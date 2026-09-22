"""Local operator configuration. No credentials or arbitrary executors are accepted."""
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .common import REPO, read_json


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_wall_seconds: int = Field(600, ge=1, le=31_536_000)
    max_disk_bytes: int = Field(2_000_000_000, ge=1, le=100_000_000_000_000)
    max_download_bytes: int = Field(0, ge=0, le=100_000_000_000_000)
    max_requests: int = Field(0, ge=0, le=100_000)
    max_epochs: int = Field(1, ge=1, le=100_000)
    max_steps: int = Field(1, ge=1, le=100_000_000)
    max_workers: int = Field(1, ge=1, le=16)
    gpu_count: int = Field(0, ge=0, le=1)
    memory_gb: float = Field(8, gt=0, le=4096)
    gpu_memory_gb: float = Field(80, gt=0, le=192)
    retries: int = Field(1, ge=0, le=3)
    max_cases: int = Field(3, ge=1, le=1_000_000)
    max_train_steps: int = Field(1, ge=1, le=100_000_000)
    max_ensemble_members: int = Field(3, ge=1, le=1000)
    device: str = "cpu"

    @field_validator("device")
    @classmethod
    def valid_device(cls, value):
        if value not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Only local cpu/cuda:0 execution is supported")
        return value


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_root: Path
    cache_root: Path
    output_root: Path
    state_root: Path
    read_roots: list[Path] = Field(default_factory=list)
    assets: dict[str, dict] = Field(default_factory=dict)
    max_concurrent_jobs: int = Field(1, ge=1, le=4)

    @field_validator("data_root", "cache_root", "output_root", "state_root")
    @classmethod
    def root_path(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("Use an absolute dedicated root, without traversal")
        value = value.expanduser().resolve()
        if value in {Path("/"), Path.home(), REPO}:
            raise ValueError("Use a dedicated subdirectory")
        return value

    @field_validator("read_roots")
    @classmethod
    def read_paths(cls, values: list[Path]) -> list[Path]:
        return [cls.root_path(v) for v in values]

    @model_validator(mode="after")
    def distinct_roots(self):
        if any(self.state_root == p or self.state_root.is_relative_to(p)
               for p in [self.data_root, self.cache_root, self.output_root, *self.read_roots]):
            raise ValueError("Approval state must be outside data/cache/output/read roots")
        return self

    @property
    def roots(self) -> list[Path]:
        return [self.data_root, self.cache_root, self.output_root, *self.read_roots]

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(read_json(path))
