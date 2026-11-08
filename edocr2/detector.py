"""The text detector: NAVER's CRAFT, from the ``craft-detector`` package, with keras-ocr's box decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .device import resolve_device
from .ocr_utils import compute_input, get_boxes, read
from .weights import CRAFT_MLT_25K, fetch


class Detector:
    """Finds word boxes, as eDOCr2's pipeline calls keras-ocr's ``Detector``.

    ``weights`` is a CRAFT state dict, by default ``craft_mlt_25k.pth`` (`edocr2.weights`).
    """

    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        from craft_detector import CRAFT

        self.device = resolve_device(device)
        model = CRAFT(pretrained=False)  # True would download torchvision's ImageNet VGG weights
        state = torch.load(weights or fetch(CRAFT_MLT_25K), map_location="cpu", weights_only=True)
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        self.model = model.to(self.device).eval()

    def predict(self, images: np.ndarray) -> np.ndarray:
        """CRAFT's text and link heatmaps (B, H/2, W/2, 2) for normalized images (B, H, W, 3)."""
        x = torch.from_numpy(np.asarray(images, dtype=np.float32)).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.inference_mode():
            return self.model(x)[0].float().cpu().numpy()

    def detect(self, images, detection_threshold=0.7, text_threshold=0.4, link_threshold=0.4, size_threshold=10):
        """Boxes (N, 4, 2) per image. An array is used as given, in whatever channel order it has."""
        inputs = np.array([compute_input(read(image)) for image in images])
        return get_boxes(
            self.predict(inputs),
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            size_threshold=size_threshold,
        )
