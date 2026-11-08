"""The text detector: NAVER's CRAFT, from the ``craft-detector`` package, with keras-ocr's box decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import resolve_device
from .ocr_utils import compute_input, get_boxes, read
from .weights import CRAFT_MLT_25K, fetch


def load_craft(weights: str | Path | None = None) -> nn.Module:
    """CRAFT with a state dict, by default ``craft_mlt_25k.pth`` (`edocr2.weights`), as built and trainable."""
    from craft_detector import CRAFT

    model = CRAFT(pretrained=False)  # True would download torchvision's ImageNet VGG weights
    state = torch.load(weights or fetch(CRAFT_MLT_25K), map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model


def _fold_batch_norms(model: nn.Module) -> nn.Module:
    """``model`` in eval mode with each batch norm folded into the convolution before it.

    In eval mode a batch norm scales and shifts each channel by constants, which the convolution's
    weights and bias can absorb: one pass over the feature map instead of two, the same function up
    to rounding. The result cannot be trained, since training normalizes by each batch's statistics.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model.eval()
    for seq in list(model.modules()):
        if isinstance(seq, nn.Sequential):
            for i in range(len(seq) - 1):
                if isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d):
                    seq[i], seq[i + 1] = fuse_conv_bn_eval(seq[i], seq[i + 1]), nn.Identity()
    return model


class Detector:
    """Finds word boxes, as eDOCr2's pipeline calls keras-ocr's ``Detector``.

    ``weights`` is a CRAFT state dict, by default ``craft_mlt_25k.pth`` (`edocr2.weights`). On the
    CPU and MPS the network has its batch norms folded. The CPU also runs it channels-last, the
    layout x86's oneDNN convolutions prefer; on MPS channels-last ran slower. CUDA, not measured,
    runs CRAFT as built. To train, use `load_craft`'s model, which keeps its batch norms.
    """

    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        self.device = resolve_device(device)
        kind = torch.device(self.device).type
        self._channels_last = kind == "cpu"
        model = load_craft(weights).eval()
        if kind in ("cpu", "mps"):
            model = _fold_batch_norms(model)
        if self._channels_last:
            model = model.to(memory_format=torch.channels_last)
        self.model = model.to(self.device)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """CRAFT's text and link heatmaps (B, H/2, W/2, 2) for normalized images (B, H, W, 3)."""
        x = torch.from_numpy(np.asarray(images, dtype=np.float32)).permute(0, 3, 1, 2)
        # The permuted (B, H, W, 3) array already has channels-last strides, so the CPU path copies nothing.
        fmt = torch.channels_last if self._channels_last else torch.contiguous_format
        x = x.contiguous(memory_format=fmt).to(self.device)
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
