"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Select one checkpoint and YAML initialization period consistently for evaluation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from finetune.checkpoint_selection import (
    refinement_checkpoint_status,
    select_refinement_checkpoint,
)


@dataclass(frozen=True)
class InferenceSelection:
    start: np.datetime64
    end: np.datetime64
    checkpoint: Path
    sha256: str
    run_id: str
    epoch: int
    validated: bool

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], config_path: Path) -> 'InferenceSelection':
        try:
            bounds = raw['data']['split_times']['test']
            start = np.datetime64(bounds['start'], 'ns')
            end = np.datetime64(bounds['end'], 'ns')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('YAML data.split_times.test must define start and end.') from exc
        if np.isnat(start) or np.isnat(end) or start > end:
            raise ValueError('YAML test dates must be finite with start <= end.')
        config_path = config_path.resolve()
        paths = raw.get('paths', {})
        root = Path(paths.get('project_root', config_path.parent)).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
        output = Path(paths.get('output_dir', root / 'outputs')).expanduser()
        if not output.is_absolute():
            output = root / output
        directory = Path(paths.get('checkpoint_dir', output / 'checkpoints')).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        case = str(raw['case_name']).strip()
        if directory.name != case:
            directory = directory / case
        directory = directory.resolve()
        initial_status = refinement_checkpoint_status(directory)
        checkpoint = select_refinement_checkpoint(directory, prefer_latest=True)
        before = checkpoint.stat()
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        after = checkpoint.stat()
        def file_version(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        if file_version(before) != file_version(after):
            raise ValueError('Checkpoint changed while being hashed; rerun evaluation after saving finishes.')
        status = refinement_checkpoint_status(directory)
        # Recheck provenance in case a new training run started during hashing.
        select_refinement_checkpoint(directory, prefer_latest=True)
        if (status['training_run_id'] != initial_status['training_run_id']
                or status['last']['metadata'] != initial_status['last']['metadata']
                or file_version(checkpoint.stat()) != file_version(before)):
            raise ValueError('Checkpoint or its provenance changed during selection; rerun evaluation after saving finishes.')
        metadata = status['last']['metadata']
        if type(metadata.get('epoch')) is not int or metadata['epoch'] < 0:
            raise ValueError('Latest checkpoint metadata must contain a nonnegative integer epoch.')
        return cls(start, end, checkpoint, digest.hexdigest(), str(status['training_run_id']),
                   int(metadata['epoch']), bool(status['last']['validated_for_inference']))

    def contains(self, initialization: Any) -> bool:
        value = np.datetime64(initialization, 'ns')
        return bool(self.start <= value <= self.end)

    def accepts(self, attrs: Mapping[str, Any]) -> bool:
        """Require the exact weights; also reject conflicting run metadata."""
        if attrs.get('checkpoint_sha256') != self.sha256:
            return False
        run = attrs.get('checkpoint_training_run_id')
        return run is None or str(run) == self.run_id

    def provenance(self) -> dict[str, Any]:
        return {
            'initialization_period_start': str(self.start),
            'initialization_period_end': str(self.end),
            'checkpoint_path': str(self.checkpoint),
            'checkpoint_sha256': self.sha256,
            'checkpoint_training_run_id': self.run_id,
            'checkpoint_epoch': self.epoch,
            'checkpoint_validated_for_inference': int(self.validated),
        }

    def expected_initializations(self, truth_times: Any, input_steps: int) -> set[int]:
        times = np.asarray(truth_times, dtype='datetime64[ns]')
        if input_steps < 1 or not times.size or np.isnat(times).any():
            raise ValueError('Evaluation needs valid truth timestamps and positive input_time_steps.')
        if np.any(times[1:] <= times[:-1]):
            raise ValueError('Truth timestamps must be strictly increasing and unique.')
        if times[0] > self.start or times[-1] < self.end:
            raise ValueError('Prepared truth does not cover the YAML test period; regenerate test.nc.')
        anchors = times[input_steps - 1:]
        return set(anchors[(anchors >= self.start) & (anchors <= self.end)].astype(np.int64).tolist())
