"""Reading one drawing end to end: segmentation, title block, GD&T frames, then dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import tools
from .detector import Detector
from .recognizer import Recognizer
from .weights import CRAFT_MLT_25K, DIMENSIONS, DIMENSIONS_ALPHABET, GDTS, GDTS_ALPHABET, fetch


@dataclass
class Models:
    detector: Detector
    dimensions: Recognizer
    gdt: Recognizer


def load_models(device: str | None = None, weights_dir: str | Path | None = None) -> Models:
    """The detector and both recognizers on ``device`` (see `resolve_device`), weights as `fetch` finds them."""

    def alphabet(weight) -> str:
        return fetch(weight, weights_dir).read_text(encoding="utf-8").splitlines()[0].strip()

    return Models(
        detector=Detector(fetch(CRAFT_MLT_25K, weights_dir), device),
        dimensions=Recognizer(fetch(DIMENSIONS, weights_dir), alphabet(DIMENSIONS_ALPHABET), device),
        gdt=Recognizer(fetch(GDTS, weights_dir), alphabet(GDTS_ALPHABET), device),
    )


@dataclass
class DrawingResult:
    """What one drawing reads as, in eDOCr2's shapes.

    ``tables``: per title block or table, Tesseract's words (``text``, ``left``, ``top``, ``width``,
    ``height``) in page pixels. ``gdt``: ``(text, (x, y))`` per GD&T frame, cells joined by ``|``,
    in page pixels. ``dimensions`` and ``other``: ``(text, box)`` with a (4, 2) box in the pixels of
    the region inside ``frame``; add ``frame.x`` and ``frame.y`` for page pixels. ``frame`` is the
    drawing border found, or None.
    """

    tables: list[list[dict[str, Any]]]
    gdt: list[tuple[str, tuple[int, int]]]
    dimensions: list[tuple[str, np.ndarray]]
    other: list[tuple[str, np.ndarray]]
    frame: Any


def read_drawing(
    img: np.ndarray,
    models: Models,
    *,
    language: str = "eng",
    autoframe: bool = True,
    frame_thres: float = 0.7,
    gdt_thres: float = 0.02,
    binary_thres: int = 127,
    cluster_thres: int = 20,
    max_img_size: int = 1048,
) -> DrawingResult:
    """Read a drawing image (BGR, as ``cv2.imread`` gives it) with eDOCr2's pipeline.

    The defaults are those of eDOCr2's ``test_drawing.py``. ``language`` is Tesseract's for the
    title block and for text that is not a dimension; Tesseract also needs ``nor`` installed, which
    the dimension filter asks for.
    """
    pipes = tools.ocr_pipelines
    _, frame, gdt_boxes, tables, dim_boxes = tools.layer_segm.segment_img(
        img, autoframe=autoframe, frame_thres=frame_thres, GDT_thres=gdt_thres, binary_thres=binary_thres
    )
    process_img = img.copy()
    table_results, _, process_img = pipes.ocr_tables(tables, process_img, language)
    gdt_results, _, process_img = pipes.ocr_gdt(process_img, gdt_boxes, models.gdt)
    if frame:
        process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
    dimensions, other, _, _ = pipes.ocr_dimensions(
        process_img,
        models.detector,
        models.dimensions,
        models.dimensions.alphabet,
        frame,
        dim_boxes,
        cluster_thres=cluster_thres,
        max_img_size=max_img_size,
        language=language,
        backg_save=False,
    )
    return DrawingResult(table_results, gdt_results, dimensions, other, frame)
