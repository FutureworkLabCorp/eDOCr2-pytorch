"""The text recognizer: eDOCr2's CRNN, read from its ``.keras`` file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .device import resolve_device
from .models import load_crnn
from .ocr_utils import read_and_fit


def read_alphabet(weights: str | Path) -> str:
    """The alphabet eDOCr2 ships beside a recognizer: the first line of the ``.txt`` of the same name."""
    with Path(weights).with_suffix(".txt").open(encoding="utf-8") as f:
        return f.readline().strip()


def _collapse(best: list[int], alphabet: str) -> str:
    """Greedy CTC decoding: repeats merged, then blanks (the last class) dropped."""
    chars, prev = [], None
    for idx in best:
        if idx != prev and idx != len(alphabet):
            chars.append(alphabet[idx])
        prev = idx
    return "".join(chars)


class Recognizer:
    """Reads one cropped line of text, as eDOCr2's pipeline calls keras-ocr's ``Recognizer``."""

    def __init__(self, weights: str | Path, alphabet: str | None = None, device: str | None = None):
        self.alphabet = read_alphabet(weights) if alphabet is None else alphabet
        self.device = resolve_device(device)
        self.model = load_crnn(weights).to(self.device)
        if self.model.n_classes != len(self.alphabet) + 1:
            raise ValueError(
                f"{weights} has {self.model.n_classes} classes; its alphabet needs {len(self.alphabet) + 1}"
            )

    def recognize(self, image: np.ndarray) -> str:
        """The text in ``image``, letterboxed onto 31 x 200 with black and read in grayscale."""
        image = read_and_fit(image, width=self.model.width, height=self.model.height, cval=0)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        x = torch.from_numpy(image.astype("float32")[np.newaxis] / 255).to(self.device)
        with torch.inference_mode():
            best = self.model(x).argmax(-1)[0].cpu().tolist()
        return _collapse(best, self.alphabet)
