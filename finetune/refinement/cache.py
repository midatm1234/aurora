"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Deterministic Aurora rollout / conditioning cache.

When Aurora is frozen, the deterministic rollout for a given
``(checkpoint, dataset, initialization time, lead time, ...)`` is a pure
function and can be reused across refinement epochs. Correctness comes first:

* the cache key includes **everything** that could change the rollout;
* a stale or incompatible entry is rejected, never silently reused;
* cached rollouts are validated against freshly computed Aurora output before
  the cache is trusted;
* caching is refused entirely when Aurora is trainable (joint fine-tuning),
  because the cached values would go stale on the first optimizer step. That is
  enforced at configuration-resolution time as well.

Adapted from ``granitewxc.refinement.cache`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), with Aurora's forecast
initialization/valid/lead times and rollout interval added to the key.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

__all__ = ["CACHE_SCHEMA_VERSION", "RolloutCache", "RolloutCacheKey"]

#: Bump whenever the cached payload layout or the rollout semantics change.
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RolloutCacheKey:
    """Everything that can change a deterministic Aurora rollout."""

    aurora_fingerprint: str
    dataset: str
    split: str
    init_time: str
    valid_time: str
    lead_time_hours: float
    rollout_interval_hours: float
    input_history_steps: int
    variables: tuple[str, ...]
    levels: tuple[float, ...]
    domain: tuple[float, float, float, float]
    normalization: str
    schema_version: int = CACHE_SCHEMA_VERSION
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutCache:
    """File-backed cache of packed deterministic rollouts.

    Entries are stored as ``<path>/<digest>.pt`` and always carry the full key
    so a mismatch can be reported precisely rather than assumed away.
    """

    def __init__(self, path: str | os.PathLike, *, enabled: bool = True) -> None:
        self.path = os.fspath(path)
        self.enabled = bool(enabled)
        if self.enabled:
            os.makedirs(self.path, exist_ok=True)

    def _file(self, key: RolloutCacheKey) -> str:
        return os.path.join(self.path, f"{key.digest()}.pt")

    def store(self, key: RolloutCacheKey, tensors: Mapping[str, torch.Tensor]) -> str | None:
        if not self.enabled:
            return None
        payload = {
            "key": key.to_dict(),
            "schema_version": CACHE_SCHEMA_VERSION,
            "tensors": {name: value.detach().to("cpu") for name, value in tensors.items()},
        }
        target = self._file(key)
        tmp = target + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, target)
        return target

    def load(self, key: RolloutCacheKey) -> dict[str, torch.Tensor] | None:
        """Return the cached tensors, or ``None`` for a miss or a stale entry."""
        if not self.enabled:
            return None
        target = self._file(key)
        if not os.path.exists(target):
            return None
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            return None
        stored = payload.get("key", {})
        if stored != key.to_dict():
            # Digest collision or a hand-edited file: refuse rather than guess.
            raise RuntimeError(
                f"Rollout cache entry {target!r} does not match the requested key. "
                "Delete the cache directory and regenerate it."
            )
        return payload.get("tensors")

    @staticmethod
    def validate(
        cached: Mapping[str, torch.Tensor],
        online: Mapping[str, torch.Tensor],
        *,
        tolerance: float = 1e-5,
    ) -> None:
        """Raise when a cached rollout disagrees with freshly computed output."""
        missing = sorted(set(online) - set(cached))
        if missing:
            raise RuntimeError(f"Cached rollout is missing tensor(s) {missing}.")
        for name, reference in online.items():
            candidate = cached[name].to(device=reference.device, dtype=reference.dtype)
            if candidate.shape != reference.shape:
                raise RuntimeError(
                    f"Cached rollout {name!r} has shape {tuple(candidate.shape)} but the "
                    f"online Aurora produced {tuple(reference.shape)}."
                )
            diff = (candidate - reference).abs()
            finite = torch.isfinite(diff)
            worst = float(diff[finite].max()) if bool(finite.any()) else 0.0
            if worst > tolerance:
                raise RuntimeError(
                    f"Cached rollout {name!r} differs from the online Aurora output by "
                    f"{worst:.3e} (> {tolerance:.3e}). The cache is stale; regenerate it."
                )


def build_cache_key(
    *,
    aurora_fingerprint: str,
    dataset: str,
    split: str,
    init_time: Any,
    valid_time: Any,
    lead_time_hours: float,
    rollout_interval_hours: float,
    input_history_steps: int,
    variables: Sequence[str],
    levels: Sequence[float],
    domain: Sequence[float],
    normalization: str,
    extra: Mapping[str, Any] | None = None,
) -> RolloutCacheKey:
    """Convenience constructor that normalises the free-form inputs."""
    if len(domain) != 4:
        raise ValueError("domain must be (lat_min, lat_max, lon_min, lon_max).")
    return RolloutCacheKey(
        aurora_fingerprint=str(aurora_fingerprint),
        dataset=str(dataset),
        split=str(split),
        init_time=str(init_time),
        valid_time=str(valid_time),
        lead_time_hours=float(lead_time_hours),
        rollout_interval_hours=float(rollout_interval_hours),
        input_history_steps=int(input_history_steps),
        variables=tuple(str(v) for v in variables),
        levels=tuple(float(v) for v in levels),
        domain=(float(domain[0]), float(domain[1]), float(domain[2]), float(domain[3])),
        normalization=str(normalization),
        extra=tuple(sorted((str(k), str(v)) for k, v in (extra or {}).items())),
    )
