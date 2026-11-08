"""On the CPU the detector folds CRAFT's batch norms without changing what CRAFT computes.

Needs ``craft_mlt_25k.pth``, fetched once into the cache or read from EDOCR2_WEIGHTS_DIR.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from edocr2 import Detector
from edocr2.detector import load_craft


def _batch_norms(model: nn.Module) -> int:
    return sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_cpu_detector_folds_batch_norms_and_keeps_the_heatmaps():
    detector = Detector(device="cpu")
    images = np.random.default_rng(0).normal(size=(2, 96, 128, 3)).astype("float32")
    with torch.inference_mode():
        expected = load_craft().eval()(torch.from_numpy(images).permute(0, 3, 1, 2).contiguous())[0].numpy()
    assert _batch_norms(detector.model) == 0
    np.testing.assert_allclose(detector.predict(images), expected, atol=1e-4)


def test_load_craft_keeps_batch_norms_to_train():
    assert _batch_norms(load_craft()) > 0
