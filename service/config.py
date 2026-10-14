"""Environment-backed configuration for the inference service."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


@dataclass(frozen=True)
class ServiceSettings:
    """Runtime settings. Paths are never included in API responses."""

    api_key: Optional[str]
    allow_empty_api_key: bool
    config_path: Optional[Path]
    checkpoint_path: Optional[Path]
    device: str
    host: str
    port: int
    max_upload_bytes: int
    model_name: str = "P2PFormer"
    model_version_override: Optional[str] = None
    protocol_version: str = "1.0"
    service_version: str = "1.0.0"
    max_concurrency: int = 1

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        config = os.getenv("P2PFORMER_CONFIG", "").strip()
        checkpoint = os.getenv("P2PFORMER_CHECKPOINT", "").strip()
        allow_empty = _parse_bool(
            os.getenv("P2PFORMER_ALLOW_EMPTY_API_KEY", "false"),
            "P2PFORMER_ALLOW_EMPTY_API_KEY",
        )
        max_upload_mb = _parse_positive_int(
            os.getenv("P2PFORMER_MAX_UPLOAD_MB", "64"),
            "P2PFORMER_MAX_UPLOAD_MB",
        )
        return cls(
            api_key=os.getenv("P2PFORMER_API_KEY"),
            allow_empty_api_key=allow_empty,
            config_path=Path(config).expanduser() if config else None,
            checkpoint_path=(
                Path(checkpoint).expanduser() if checkpoint else None
            ),
            device=os.getenv("P2PFORMER_DEVICE", "cuda:0").strip(),
            host=os.getenv("P2PFORMER_HOST", "127.0.0.1").strip(),
            port=_parse_positive_int(
                os.getenv("P2PFORMER_PORT", "8000"), "P2PFORMER_PORT"
            ),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            model_version_override=(
                os.getenv("P2PFORMER_MODEL_VERSION", "").strip() or None
            ),
        )

    @property
    def config_exists(self) -> bool:
        return bool(self.config_path and self.config_path.is_file())

    @property
    def checkpoint_exists(self) -> bool:
        return bool(self.checkpoint_path and self.checkpoint_path.is_file())

    @property
    def model_version(self) -> str:
        if self.model_version_override:
            return self.model_version_override
        if self.checkpoint_path:
            return self.checkpoint_path.stem
        return "unavailable"

    @property
    def display_device(self) -> str:
        if self.device.lower().startswith("cuda"):
            return "CUDA"
        return self.device.upper()


def cuda_is_available() -> bool:
    """Return the real CUDA state without making torch a service import dep."""

    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False

