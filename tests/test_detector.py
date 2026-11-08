"""On the CPU and MPS the detector folds CRAFT's batch norms without changing what CRAFT computes.

Needs ``craft_mlt_25k.pth``, fetched once into the cache or read from EDOCR2_WEIGHTS_DIR. Checks
MPS as well when present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from edocr2 import Detector
from edocr2.detector import load_craft


def _batch_norms(model: nn.Module) -> int:
    return sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())


@pytest.mark.parametrize("device", ["cpu"] + ["mps"] * torch.backends.mps.is_available())
def test_detector_folds_batch_norms_and_keeps_the_heatmaps(device):
    detector = Detector(device=device)
    images = np.random.default_rng(0).normal(size=(2, 96, 128, 3)).astype("float32")
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
    with torch.inference_mode():
        expected = load_craft().eval().to(device)(x)[0].cpu().numpy()
    assert _batch_norms(detector.model) == 0
    np.testing.assert_allclose(detector.predict(images), expected, atol=1e-4)


def test_load_craft_keeps_batch_norms_to_train():
    assert _batch_norms(load_craft()) > 0
