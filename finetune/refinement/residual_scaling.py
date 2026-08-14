"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Residual standardization for the Phase-2 stochastic refiners.

Phase 2 builds its residual in Aurora's normalized target space:

    r = target_norm - rollout_norm

Aurora's field normalization does not make that forecast residual unit scale.
This module therefore gives stochastic refiners an invertible boundary:

    z = (r - shift) / scale
    r_hat = z_hat * scale + shift

Only active scaling modes persist statistics. The identity mode deliberately has
an empty state dict so legacy no-scaling checkpoints remain strictly loadable.

Production training uses an explicit calibration pass over the complete
training split. Each rank accumulates disjoint local count/sum/sum-of-squares
statistics and a single collective reduction is performed only after that pass.
The completed statistics, method, counts, and split fingerprint are checkpointed
and frozen before the first optimiser step. The older online warm-up API remains
available for checkpoint compatibility and small standalone experiments, but is
not the production path.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

__all__ = ["ResidualScaler"]

_MIN_SCALE = 1.0e-8
_FINGERPRINT_BYTES = 32

_CALIBRATION_METHOD_UNKNOWN = 0
_CALIBRATION_METHOD_ONLINE = 1
_CALIBRATION_METHOD_EXPLICIT_FIT = 2
_CALIBRATION_METHOD_TRAINING_SPLIT = 3


class ResidualScaler(nn.Module):
    """Invertible per-channel standardization of the refinement residual.

    Args:
        num_channels: Packed residual channel count.
        mode: "none" (identity), "global" (one statistic shared by all
            channels), or "per_channel" (one statistic per packed channel).
        center: Remove the residual mean in addition to scaling its standard
            deviation.
        momentum: Retained for legacy configuration/checkpoint compatibility;
            fixed cumulative statistics no longer depend on batch-order EMA.
        warmup_batches: Number of non-empty training batches accumulated before
            fixed statistics are frozen.
        target_std: Desired standard deviation in generative space.
        max_scale / min_scale: Bounds for non-degenerate active channels.
    """

    MODES = ("none", "global", "per_channel")

    def __init__(
        self,
        num_channels: int,
        *,
        mode: str = "per_channel",
        center: bool = False,
        momentum: float = 0.05,
        warmup_batches: int = 32,
        target_std: float = 1.0,
        min_scale: float = 1.0e-4,
        max_scale: float = 1.0e4,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(
                f"Unsupported residual scaling mode {mode!r}; expected {self.MODES}."
            )
        if int(num_channels) <= 0:
            raise ValueError(
                "ResidualScaler num_channels must be a positive integer; "
                f"got {num_channels!r}."
            )
        if int(warmup_batches) < 0:
            raise ValueError(
                "ResidualScaler warmup_batches must be non-negative; "
                f"got {warmup_batches!r}."
            )
        if not 0.0 <= float(momentum) <= 1.0:
            raise ValueError(
                "ResidualScaler momentum must be in [0, 1]; "
                f"got {momentum!r}."
            )
        if float(target_std) <= 0.0:
            raise ValueError(
                "ResidualScaler target_std must be positive; "
                f"got {target_std!r}."
            )
        if float(min_scale) <= 0.0 or float(max_scale) < float(min_scale):
            raise ValueError(
                "ResidualScaler requires 0 < min_scale <= max_scale; "
                f"got min_scale={min_scale!r}, max_scale={max_scale!r}."
            )

        self.mode = mode
        self.center = bool(center)
        self.momentum = float(momentum)
        self.warmup_batches = int(warmup_batches)
        self.target_std = float(target_std)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.num_channels = int(num_channels)

        # These are runtime controls rather than model state. Persistent
        # sufficient statistics below make an interrupted calibration resumable.
        self._distributed_calibration = False
        self._distributed_freeze_after = max(1, self.warmup_batches)
        self._exact_calibration_active = False

        shape = (1, self.num_channels, 1, 1)
        # Identity mode keeps non-persistent identity buffers for API
        # compatibility, while contributing no keys to the parent state dict.
        persistent = self.is_active
        self.register_buffer("scale", torch.ones(shape), persistent=persistent)
        self.register_buffer("shift", torch.zeros(shape), persistent=persistent)
        self.register_buffer(
            "observed_batches",
            torch.zeros((), dtype=torch.long),
            persistent=persistent,
        )
        self.register_buffer(
            "observed_channels",
            torch.zeros(shape, dtype=torch.long),
            persistent=persistent,
        )
        self.register_buffer(
            "frozen", torch.zeros((), dtype=torch.bool), persistent=persistent
        )
        # Float64 accumulation avoids cancellation when a small residual mean
        # is subtracted from its second moment.
        self.register_buffer(
            "calibration_count",
            torch.zeros(shape, dtype=torch.float64),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_sum",
            torch.zeros(shape, dtype=torch.float64),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_sum_sq",
            torch.zeros(shape, dtype=torch.float64),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_complete",
            torch.zeros((), dtype=torch.bool),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_method",
            torch.zeros((), dtype=torch.long),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_examples",
            torch.zeros((), dtype=torch.long),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_logical_samples",
            torch.zeros((), dtype=torch.long),
            persistent=persistent,
        )
        self.register_buffer(
            "calibration_fingerprint",
            torch.zeros(_FINGERPRINT_BYTES, dtype=torch.uint8),
            persistent=persistent,
        )

    # -- estimation ------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self.mode != "none"

    @property
    def is_fitted(self) -> bool:
        return self.is_active and bool(
            (self.calibration_count > 0).all().item()
        )

    @property
    def is_ready(self) -> bool:
        """Whether a fixed, complete transform is safe for evaluation."""
        return (
            not self.is_active
            or (
                self.is_fitted
                and bool(self.frozen.item())
                and bool(self.calibration_complete.item())
            )
        )

    @property
    def has_exact_training_split_calibration(self) -> bool:
        return self.is_ready and int(self.calibration_method.item()) == (
            _CALIBRATION_METHOD_TRAINING_SPLIT
        )

    @property
    def calibration_fingerprint_hex(self) -> str:
        if not self.is_active:
            return ""
        return bytes(self.calibration_fingerprint.detach().cpu().tolist()).hex()

    @property
    def distributed_calibration_enabled(self) -> bool:
        return self.is_active and self._distributed_calibration

    def train(self, mode: bool = True):  # noqa: D102 - torch API
        if not mode and self.is_active and not self.is_ready:
            raise RuntimeError(
                "Active ResidualScaler cannot enter eval/inference mode before "
                "finite training correction statistics are complete and frozen. "
                "Run the training-split calibration pre-pass or call fit() on "
                "training-only corrections first."
            )
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Migrate checkpoints written before calibration provenance buffers.

        Legacy active scalers retain their exact saved scale/shift and sufficient
        statistics. A fitted and frozen legacy scaler is marked complete but its
        method remains ``online``; production resume code can therefore preserve
        its numerical transform while refusing to mislabel it as a complete-split
        calibration.
        """
        if self.is_active:
            frozen = state_dict.get(prefix + "frozen")
            count = state_dict.get(prefix + "calibration_count")
            legacy_complete = bool(
                frozen is not None
                and bool(torch.as_tensor(frozen).item())
                and count is not None
                and bool((torch.as_tensor(count) > 0).all().item())
            )
            defaults = {
                "calibration_complete": torch.tensor(
                    legacy_complete, dtype=self.calibration_complete.dtype
                ),
                "calibration_method": torch.tensor(
                    _CALIBRATION_METHOD_ONLINE
                    if legacy_complete
                    else _CALIBRATION_METHOD_UNKNOWN,
                    dtype=self.calibration_method.dtype,
                ),
                "calibration_examples": torch.zeros(
                    (), dtype=self.calibration_examples.dtype
                ),
                "calibration_logical_samples": torch.zeros(
                    (), dtype=self.calibration_logical_samples.dtype
                ),
                "calibration_fingerprint": torch.zeros(
                    _FINGERPRINT_BYTES, dtype=self.calibration_fingerprint.dtype
                ),
            }
            for name, value in defaults.items():
                state_dict.setdefault(prefix + name, value)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def freeze(self, value: bool = True) -> None:
        """Stop or resume updating active running statistics."""
        if self.is_active:
            self.frozen.fill_(bool(value))

    def enable_distributed_calibration(
        self, *, freeze_after_batches: int | None = None
    ) -> None:
        """Synchronize calibration across a manually distributed trainer.

        This must be called on every rank after the process group is initialized
        and before the first training forward. Every subsequent training
        observation performs one small all-reduce of per-channel sufficient
        statistics. Calibration freezes after freeze_after_batches globally
        non-empty observations. The threshold defaults to warmup_batches, with
        a minimum of one.
        """
        if not self.is_active:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ResidualScaler distributed calibration requires an initialized "
                "torch.distributed process group."
            )
        threshold = (
            self.warmup_batches
            if freeze_after_batches is None
            else int(freeze_after_batches)
        )
        if threshold <= 0:
            raise ValueError(
                "ResidualScaler distributed freeze_after_batches must be positive; "
                f"got {threshold!r}."
            )
        self._distributed_freeze_after = threshold
        self._distributed_calibration = dist.get_world_size() > 1
        if (
            self._distributed_calibration
            and int(self.observed_batches.item()) >= threshold
        ):
            self.freeze()

    def _validate_inputs(
        self, correction: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not correction.is_floating_point():
            raise TypeError(
                "ResidualScaler correction must have a floating dtype, got "
                f"{correction.dtype}."
            )
        values = correction.detach().float()
        if values.ndim != 4:
            raise ValueError(
                "ResidualScaler correction must have shape [N, C, H, W]; "
                f"got {tuple(values.shape)}."
            )
        if values.shape[1] != self.num_channels:
            raise ValueError(
                "ResidualScaler correction channel dimension must equal "
                f"num_channels={self.num_channels}; got shape {tuple(values.shape)}."
            )
        finite = torch.isfinite(values)
        if mask is None:
            valid = finite
        else:
            supplied = mask.detach().to(device=values.device, dtype=torch.bool)
            try:
                supplied = torch.broadcast_to(supplied, values.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "ResidualScaler mask must be broadcastable to correction shape "
                    f"{tuple(values.shape)}; got {tuple(mask.shape)}."
                ) from exc
            valid = supplied & finite
        return values, valid

    def _sufficient_statistics(
        self, residual: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, valid = self._validate_inputs(residual, mask)
        clean = torch.where(valid, values, torch.zeros((), device=values.device))
        clean64 = clean.double()
        if self.mode == "per_channel":
            dims = (0, 2, 3)
            count = valid.sum(dim=dims, keepdim=True, dtype=torch.float64)
            value_sum = clean64.sum(dim=dims, keepdim=True)
            value_sum_sq = clean64.square().sum(dim=dims, keepdim=True)
        else:
            dims = (0, 1, 2, 3)
            count = valid.sum(dim=dims, keepdim=True, dtype=torch.float64)
            value_sum = clean64.sum(dim=dims, keepdim=True)
            value_sum_sq = clean64.square().sum(dim=dims, keepdim=True)
            shape = (1, self.num_channels, 1, 1)
            count = count.expand(shape).clone()
            value_sum = value_sum.expand(shape).clone()
            value_sum_sq = value_sum_sq.expand(shape).clone()
        return count, value_sum, value_sum_sq

    @staticmethod
    def _distributed_sum(
        count: torch.Tensor, value_sum: torch.Tensor, value_sum_sq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = count.shape
        width = count.numel()
        packed = torch.cat(
            (count.reshape(-1), value_sum.reshape(-1), value_sum_sq.reshape(-1))
        )
        backend = str(dist.get_backend()).lower()
        if "nccl" not in backend:
            packed_for_reduce = packed.cpu()
        else:
            packed_for_reduce = packed
        dist.all_reduce(packed_for_reduce, op=dist.ReduceOp.SUM)
        if packed_for_reduce.device != packed.device:
            packed = packed_for_reduce.to(packed.device)
        else:
            packed = packed_for_reduce
        return (
            packed[:width].reshape(shape),
            packed[width : 2 * width].reshape(shape),
            packed[2 * width :].reshape(shape),
        )

    @torch.no_grad()
    def begin_exact_training_split_calibration(self) -> None:
        """Start deferred local accumulation over the complete train split."""
        if not self.is_active:
            return
        if not self.training:
            raise RuntimeError(
                "Exact residual calibration must run with the scaler in training mode."
            )
        self._distributed_calibration = False
        self._exact_calibration_active = True
        self.frozen.fill_(False)
        self.calibration_complete.fill_(False)
        self.calibration_method.fill_(_CALIBRATION_METHOD_UNKNOWN)
        self.observed_batches.zero_()
        self.observed_channels.zero_()
        self.calibration_count.zero_()
        self.calibration_sum.zero_()
        self.calibration_sum_sq.zero_()
        self.calibration_examples.zero_()
        self.calibration_logical_samples.zero_()
        self.calibration_fingerprint.zero_()

    @torch.no_grad()
    def abort_exact_training_split_calibration(self) -> None:
        """Leave an interrupted calibration unusable by evaluation/inference."""
        if not self.is_active:
            return
        self._exact_calibration_active = False
        self.frozen.fill_(False)
        self.calibration_complete.fill_(False)
        self.calibration_method.fill_(_CALIBRATION_METHOD_UNKNOWN)

    @staticmethod
    def _fingerprint_bytes(value: bytes | bytearray | str) -> bytes:
        if isinstance(value, str):
            try:
                value = bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(
                    "Residual calibration fingerprint must be hexadecimal."
                ) from exc
        result = bytes(value)
        if len(result) != _FINGERPRINT_BYTES:
            raise ValueError(
                "Residual calibration fingerprint must contain exactly "
                f"{_FINGERPRINT_BYTES} bytes; got {len(result)}."
            )
        return result

    @torch.no_grad()
    def _reduce_exact_statistics_once(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Distributed exact residual calibration requires an initialized "
                "torch.distributed process group."
            )
        if dist.get_world_size() <= 1:
            return
        shape = self.calibration_count.shape
        width = self.calibration_count.numel()
        packed = torch.cat(
            (
                self.calibration_count.reshape(-1),
                self.calibration_sum.reshape(-1),
                self.calibration_sum_sq.reshape(-1),
                self.observed_channels.double().reshape(-1),
                self.observed_batches.double().reshape(1),
                self.calibration_examples.double().reshape(1),
            )
        )
        backend = str(dist.get_backend()).lower()
        reduced = packed if "nccl" in backend else packed.cpu()
        # The sole collective in the calibration pass: local shards may be
        # unequal and are never padded with duplicate training examples.
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if reduced.device != packed.device:
            reduced = reduced.to(packed.device)
        offset = 0

        def take(size: int) -> torch.Tensor:
            nonlocal offset
            result = reduced[offset : offset + size]
            offset += size
            return result

        self.calibration_count.copy_(take(width).reshape(shape))
        self.calibration_sum.copy_(take(width).reshape(shape))
        self.calibration_sum_sq.copy_(take(width).reshape(shape))
        self.observed_channels.copy_(
            take(width).reshape(shape).to(dtype=self.observed_channels.dtype)
        )
        self.observed_batches.copy_(
            take(1).reshape(()).to(dtype=self.observed_batches.dtype)
        )
        self.calibration_examples.copy_(
            take(1).reshape(()).to(dtype=self.calibration_examples.dtype)
        )

    @torch.no_grad()
    def finalize_exact_training_split_calibration(
        self,
        *,
        logical_samples: int,
        fingerprint: bytes | bytearray | str,
        packed_examples: int | None = None,
        distributed: bool = False,
    ) -> None:
        """Reduce once, validate every channel, materialize, and freeze stats."""
        if not self.is_active:
            return
        if not self._exact_calibration_active:
            raise RuntimeError("No exact residual calibration session is active.")
        if int(logical_samples) <= 0:
            raise ValueError(
                "Exact residual calibration requires at least one logical sample."
            )
        fingerprint_bytes = self._fingerprint_bytes(fingerprint)
        try:
            if distributed:
                self._reduce_exact_statistics_once()
            for name, statistic in (
                ("count", self.calibration_count),
                ("sum", self.calibration_sum),
                ("sum of squares", self.calibration_sum_sq),
            ):
                if not bool(torch.isfinite(statistic).all().item()):
                    raise RuntimeError(
                        "Exact training-split residual calibration produced "
                        f"non-finite cumulative {name}."
                    )
            if not bool((self.calibration_count > 0).all().item()):
                missing = torch.nonzero(
                    self.calibration_count.reshape(-1) <= 0
                ).reshape(-1).tolist()
                raise RuntimeError(
                    "Exact training-split residual calibration has no finite, valid "
                    f"cells for packed channel indices {missing}."
                )
            if packed_examples is not None and int(
                self.calibration_examples.item()
            ) != int(packed_examples):
                raise RuntimeError(
                    "Residual calibration packed-example count mismatch: observed="
                    f"{int(self.calibration_examples.item())}, expected="
                    f"{int(packed_examples)}."
                )
            self._set_from_cumulative_statistics()
            self.calibration_logical_samples.fill_(int(logical_samples))
            self.calibration_fingerprint.copy_(
                torch.tensor(
                    list(fingerprint_bytes),
                    device=self.calibration_fingerprint.device,
                    dtype=self.calibration_fingerprint.dtype,
                )
            )
            self.calibration_method.fill_(_CALIBRATION_METHOD_TRAINING_SPLIT)
            self.calibration_complete.fill_(True)
            self.frozen.fill_(True)
            self._exact_calibration_active = False
            self.validate_exact_training_split_calibration(
                logical_samples=logical_samples,
                fingerprint=fingerprint_bytes,
                packed_examples=packed_examples,
            )
        except Exception:
            self.abort_exact_training_split_calibration()
            raise

    def validate_exact_training_split_calibration(
        self,
        *,
        logical_samples: int,
        fingerprint: bytes | bytearray | str,
        packed_examples: int | None = None,
    ) -> None:
        """Validate checkpointed completion/count/split provenance metadata."""
        if not self.has_exact_training_split_calibration:
            raise RuntimeError(
                "ResidualScaler does not contain a complete exact training-split "
                "calibration. Legacy online warm-up statistics cannot be relabeled "
                "as complete-split statistics."
            )
        if int(self.calibration_logical_samples.item()) != int(logical_samples):
            raise RuntimeError(
                "Residual calibration training-sample count mismatch: checkpoint="
                f"{int(self.calibration_logical_samples.item())}, current="
                f"{int(logical_samples)}."
            )
        if packed_examples is not None and int(
            self.calibration_examples.item()
        ) != int(packed_examples):
            raise RuntimeError(
                "Residual calibration packed-example count mismatch: checkpoint="
                f"{int(self.calibration_examples.item())}, current="
                f"{int(packed_examples)}."
            )
        expected = self._fingerprint_bytes(fingerprint).hex()
        if self.calibration_fingerprint_hex != expected:
            raise RuntimeError(
                "Residual calibration training-split fingerprint mismatch: "
                f"checkpoint={self.calibration_fingerprint_hex}, current={expected}."
            )
        self._validated_statistics(self.scale)

    def raw_statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return count, mean and population std in normalized correction space."""
        count = self.calibration_count.detach().double()
        safe_count = count.clamp_min(1.0)
        mean = self.calibration_sum.detach().double() / safe_count
        variance = (
            self.calibration_sum_sq.detach().double() / safe_count - mean.square()
        ).clamp_min(0.0)
        return count, mean, variance.sqrt()

    @torch.no_grad()
    def _set_from_cumulative_statistics(self) -> None:
        valid = self.calibration_count > 0
        safe_count = self.calibration_count.clamp_min(1.0)
        mean = self.calibration_sum / safe_count
        variance = (
            self.calibration_sum_sq / safe_count - mean.square()
        ).clamp_min(0.0)
        batch_scale = (
            variance.sqrt() / max(self.target_std, _MIN_SCALE)
        ).clamp(self.min_scale, self.max_scale)
        batch_shift = mean if self.center else torch.zeros_like(mean)
        self.scale.copy_(
            torch.where(valid, batch_scale.to(self.scale.dtype), self.scale)
        )
        self.shift.copy_(
            torch.where(valid, batch_shift.to(self.shift.dtype), self.shift)
        )

    @torch.no_grad()
    def observe(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Update statistics from one training-only residual batch.

        All-masked batches do not advance calibration. Channels with zero valid
        values retain their exact previous scale, shift, and observation count.
        """
        if not self.is_active or bool(self.frozen.item()) or not self.training:
            return

        count, value_sum, value_sum_sq = self._sufficient_statistics(residual, mask)
        if self._exact_calibration_active:
            self._distributed_calibration = False
            # Exact coverage counts every packed example, including examples
            # whose cells are all masked; valid-cell statistics remain excluded.
            self.calibration_examples.add_(int(residual.shape[0]))
        if self._distributed_calibration:
            count, value_sum, value_sum_sq = self._distributed_sum(
                count, value_sum, value_sum_sq
            )

        valid = count > 0
        if not bool(valid.any().item()):
            return
        if not self._exact_calibration_active:
            self.calibration_examples.add_(int(residual.shape[0]))

        self.calibration_count.add_(count.to(self.calibration_count.device))
        self.calibration_sum.add_(value_sum.to(self.calibration_sum.device))
        self.calibration_sum_sq.add_(value_sum_sq.to(self.calibration_sum_sq.device))
        if self._exact_calibration_active:
            self.observed_channels.add_(
                valid.to(device=self.observed_channels.device, dtype=torch.long)
            )
            self.observed_batches.add_(1)
            return

        # Exact cumulative statistics are used on one rank and many ranks alike;
        # per-batch EMA makes the deployed normalizer depend on data order.
        self._set_from_cumulative_statistics()

        self.observed_channels.add_(
            valid.to(device=self.observed_channels.device, dtype=torch.long)
        )
        self.observed_batches.add_(1)
        threshold = (
            self._distributed_freeze_after
            if self._distributed_calibration
            else max(1, self.warmup_batches)
        )
        complete = bool((self.calibration_count > 0).all().item())
        if complete and int(self.observed_batches.item()) >= threshold:
            self.freeze()
            self.calibration_complete.fill_(True)
            self.calibration_method.fill_(_CALIBRATION_METHOD_ONLINE)

    @torch.no_grad()
    def fit(
        self,
        residual: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        freeze_after_fit: bool = True,
    ) -> None:
        """Fit exact training statistics and freeze them when every channel exists."""
        if not self.is_active:
            return
        was_training = self.training
        was_frozen = bool(self.frozen.item())
        self.train(True)
        self._exact_calibration_active = False
        self.frozen.fill_(False)
        self.scale.fill_(1.0)
        self.shift.zero_()
        self.observed_batches.zero_()
        self.observed_channels.zero_()
        self.calibration_count.zero_()
        self.calibration_sum.zero_()
        self.calibration_sum_sq.zero_()
        self.calibration_complete.fill_(False)
        self.calibration_method.fill_(_CALIBRATION_METHOD_UNKNOWN)
        self.calibration_examples.zero_()
        self.calibration_logical_samples.zero_()
        self.calibration_fingerprint.zero_()
        try:
            self.observe(residual, mask)
        finally:
            complete = bool((self.calibration_count > 0).all().item())
            should_freeze = complete and (was_frozen or freeze_after_fit)
            self.frozen.fill_(should_freeze)
            self.calibration_complete.fill_(should_freeze)
            if should_freeze:
                self.calibration_method.fill_(_CALIBRATION_METHOD_EXPLICIT_FIT)
            super().train(was_training)

    # -- transforms ------------------------------------------------------
    def _validate_transform(self, tensor: torch.Tensor, field_name: str) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.num_channels:
            raise ValueError(
                f"ResidualScaler {field_name} must be [N, {self.num_channels}, H, W], "
                f"got {tuple(tensor.shape)}."
            )
        if not tensor.is_floating_point():
            raise TypeError(
                f"ResidualScaler {field_name} must have a floating dtype, got "
                f"{tensor.dtype}."
            )
        value32 = tensor.float()
        if not bool(torch.isfinite(value32).all()):
            raise ValueError(f"ResidualScaler {field_name} contains non-finite values.")
        return value32

    def _validated_statistics(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training and not self.is_ready:
            raise RuntimeError(
                "Active ResidualScaler is unfitted or unfrozen; evaluation and "
                "inference require checkpointed training-only statistics."
            )
        scale = self.scale.to(device=reference.device, dtype=torch.float32)
        shift = self.shift.to(device=reference.device, dtype=torch.float32)
        if not bool(torch.isfinite(scale).all()) or bool((scale < _MIN_SCALE).any()):
            raise RuntimeError(
                "ResidualScaler checkpoint contains non-finite or near-zero scales."
            )
        if not bool(torch.isfinite(shift).all()):
            raise RuntimeError("ResidualScaler checkpoint contains non-finite shifts.")
        return scale, shift

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a normalized correction into generative space in float32."""
        value32 = self._validate_transform(residual, "correction")
        if not self.is_active:
            return value32
        scale, shift = self._validated_statistics(value32)
        return (value32 - shift) / scale

    def decode(self, scaled: torch.Tensor) -> torch.Tensor:
        """Apply the exact inverse scaling transform in float32."""
        value32 = self._validate_transform(scaled, "scaled correction")
        if not self.is_active:
            return value32
        scale, shift = self._validated_statistics(value32)
        return value32 * scale + shift

    def encode_difference(self, correction_difference: torch.Tensor) -> torch.Tensor:
        """Scale an innovation without applying the correction mean shift."""
        value32 = self._validate_transform(
            correction_difference, "correction difference"
        )
        if not self.is_active:
            return value32
        scale, _ = self._validated_statistics(value32)
        return value32 / scale

    def decode_difference(self, scaled_difference: torch.Tensor) -> torch.Tensor:
        """Inverse-scale an innovation without adding the correction mean shift."""
        value32 = self._validate_transform(
            scaled_difference, "scaled correction difference"
        )
        if not self.is_active:
            return value32
        scale, _ = self._validated_statistics(value32)
        return value32 * scale

    def decode_mask_scale(self, reference: torch.Tensor) -> torch.Tensor:
        """Return the float32 scaled-to-normalized per-channel magnitude."""
        if not self.is_active:
            return torch.ones((), device=reference.device, dtype=torch.float32)
        scale, _ = self._validated_statistics(reference)
        return scale

    def extra_repr(self) -> str:
        if not self.is_active:
            return f"mode={self.mode}, channels={self.num_channels}"
        return (
            f"mode={self.mode}, channels={self.num_channels}, center={self.center}, "
            f"fitted={self.is_fitted}, ready={self.is_ready}, "
            f"frozen={bool(self.frozen.item())}, "
            f"calibration_method={int(self.calibration_method.item())}, "
            f"distributed_calibration={self._distributed_calibration}"
        )
