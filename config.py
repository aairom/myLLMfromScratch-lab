"""Central configuration loaded from the environment (``.env``).

No value is hard-coded elsewhere in the project: every tunable path, port or
default comes from here, and every variable is documented in ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env if it exists; real environment variables always win.
load_dotenv(PROJECT_ROOT / ".env", override=False)

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    app_port: int
    device: str
    default_seed: int
    input_dir: Path
    output_dir: Path


def _resolve_dir(value: str) -> Path:
    """Resolve a folder setting relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    """Read settings from the environment, validating each value."""
    port = int(os.environ.get("APP_PORT", "8601"))
    if port == 5000:
        raise ValueError("Port 5000 is reserved by AirDrop on macOS; choose another APP_PORT.")
    device = os.environ.get("DEVICE", "auto").lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"DEVICE must be one of {VALID_DEVICES}, got {device!r}")
    return Settings(
        app_port=port,
        device=device,
        default_seed=int(os.environ.get("DEFAULT_SEED", "42")),
        input_dir=_resolve_dir(os.environ.get("INPUT_DIR", "input")),
        output_dir=_resolve_dir(os.environ.get("OUTPUT_DIR", "output")),
    )


def resolve_device(requested: str = "auto"):
    """Return a ``torch.device`` for the requested device name.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An unavailable explicit
    choice falls back to CPU so the app always runs.
    """
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return torch.device(requested)
