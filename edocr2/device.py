"""Choosing the torch device the networks run on."""

from __future__ import annotations

import os


def resolve_device(device: str | None = None) -> str:
    """``device``, else ``EDOCR2_DEVICE``, else CUDA, then MPS, then the CPU, whichever is present."""
    import torch

    chosen = device or os.environ.get("EDOCR2_DEVICE")
    if chosen:
        return chosen
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
