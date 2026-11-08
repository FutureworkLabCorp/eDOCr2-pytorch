"""`get_boxes` must decode heatmaps exactly as keras-ocr's ``getBoxes`` did.

The regression test covers real heatmaps; these random ones add the edge cases: components on the
map's edges, blobs close to square (keras-ocr's diamond case), and link areas that cut words.
"""

import cv2
import numpy as np
import pytest
from reference_boxes import reference_get_boxes

from edocr2.ocr_utils import get_boxes


def _heatmap(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    text = np.zeros((h, w), np.float32)
    link = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(5, 60)):
        y, x = rng.integers(-5, h), rng.integers(-5, w)
        bh, bw = rng.integers(2, 30), rng.integers(2, 60)
        text[max(y, 0) : y + bh, max(x, 0) : x + bw] = rng.uniform(0.3, 1.0)
        if rng.random() < 0.5:  # a link reaching out of the word, partly over it
            link[max(y, 0) : y + bh, max(x + bw // 2, 0) : x + bw + rng.integers(0, 20)] = rng.uniform(0.3, 1.0)
        if rng.random() < 0.4:  # a slanted word
            corners = cv2.boxPoints(((float(x), float(y)), (float(bw), float(bh)), float(rng.uniform(5, 85))))
            cv2.fillPoly(text, [corners.astype(np.int32)], float(rng.uniform(0.3, 1.0)))
    text += rng.uniform(0, 0.45, (h, w)).astype(np.float32) * (rng.random((h, w)) < 0.02)  # speckle
    return np.stack([text, link], -1)[np.newaxis]


@pytest.mark.parametrize("seed", range(40))
def test_matches_keras_ocr(seed):
    rng = np.random.default_rng(seed)
    y = _heatmap(rng, int(rng.integers(40, 300)), int(rng.integers(40, 300)))
    (expected,), (actual,) = reference_get_boxes(y), get_boxes(y)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
