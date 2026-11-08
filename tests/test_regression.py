"""eDOCr2-pytorch must read the seven sample drawings as eDOCr2 did on Keras.

``golden/*.json`` holds eDOCr2's output for each sample: eDOCr2 f6f9651 with its v1.0.0
recognizers on TensorFlow 2.18.1 on the CPU, which a PyTorch port of both networks reproduced
string for string and box for box (see golden/README.md).

What the networks decide must match anywhere: the GD&T frames, the dimensions and the boxes of
the other text. A box matches when it has the same four corners in the same cyclic order; which
corner comes first is shapely's floating-point choice and differs between x86 and ARM. What
Tesseract reads, the title blocks and the other text, depends on its build: it is compared only
under the Tesseract the outputs were read with (`TESSERACT`, with tessdata_fast 4.1.0's eng and
nor). On Ubuntu's Tesseract 5.5.0 the networks' outputs match and a few title block characters
do not.

Needs Tesseract with ``eng`` and ``nor`` (TESSDATA_PREFIX may point at them) and the weights,
fetched once into the cache or read from EDOCR2_WEIGHTS_DIR. Runs on the CPU, and on CUDA or MPS
as well when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

SAMPLES = Path(__file__).parent / "test_samples"
GOLDEN = Path(__file__).parent / "golden"
NAMES = sorted(p.stem for p in GOLDEN.glob("*.json"))
TESSERACT = "5.5.4"
_READ: dict[tuple[str, str], dict[str, list[str]]] = {}


def _devices() -> list[str]:
    import torch

    return ["cpu"] + ["cuda"] * torch.cuda.is_available() + ["mps"] * torch.backends.mps.is_available()


@pytest.fixture(scope="module", params=_devices())
def models(request):
    pytesseract = pytest.importorskip("pytesseract")
    try:
        languages = pytesseract.get_languages()
    except pytesseract.TesseractNotFoundError:
        pytest.skip("Tesseract is not installed")
    if not {"eng", "nor"} <= set(languages):
        pytest.skip("Tesseract needs eng and nor traineddata (set TESSDATA_PREFIX)")
    import edocr2

    return edocr2.load_models(device=request.param)


def _box(entry: str) -> str:
    """``text @x0,y0,...`` with its corners turned to start at the smallest, cyclic order kept."""
    text, _, coords = entry.rpartition(" @")
    values = [int(v) for v in coords.split(",")]
    corners = [tuple(values[i : i + 2]) for i in range(0, len(values), 2)]
    start = corners.index(min(corners))
    corners = corners[start:] + corners[:start]
    return f"{text} @{','.join(f'{x},{y}' for x, y in corners)}"


def _read(models, name: str) -> dict[str, list[str]]:
    key = (models.detector.device, name)
    if key not in _READ:
        import edocr2

        image = next(p for p in SAMPLES.glob(f"{name}.*") if p.suffix in (".jpg", ".png"))
        result = edocr2.read_drawing(cv2.imread(str(image)), models)

        def at(text, box) -> str:
            return f"{text} @{','.join(str(int(v)) for v in box.reshape(-1))}"

        _READ[key] = {
            "tables": [" ".join(word["text"] for word in table) for table in result.tables],
            "gdt": sorted(f"{text} @{x},{y}" for text, (x, y) in result.gdt),
            "dimensions": sorted(at(text, box) for text, box in result.dimensions),
            "other": sorted(at(text, box) for text, box in result.other),
        }
    return _READ[key]


def _differences(expected: dict[str, list[str]], out: dict[str, list[str]]) -> list[str]:
    return [
        f"{key}: missing {sorted(set(value) - set(out[key]))}, extra {sorted(set(out[key]) - set(value))}"
        for key, value in expected.items()
        if out[key] != value
    ]


@pytest.mark.parametrize("name", NAMES)
def test_networks_match_edocr2(models, name):
    out = _read(models, name)
    golden = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))

    def networks(read: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            "gdt": read["gdt"],
            "dimensions": sorted(map(_box, read["dimensions"])),
            "other boxes": sorted(_box(entry).rpartition(" @")[2] for entry in read["other"]),
        }

    differences = _differences(networks(golden), networks(out))
    assert not differences, "\n".join(differences)


@pytest.mark.parametrize("name", NAMES)
def test_tesseract_text_matches_edocr2(models, name):
    import pytesseract

    version = str(pytesseract.get_tesseract_version())
    if version != TESSERACT:
        pytest.skip(f"the outputs were read with Tesseract {TESSERACT}, this is {version}")
    out = _read(models, name)
    golden = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))

    def text(read: dict[str, list[str]]) -> dict[str, list[str]]:
        return {"tables": read["tables"], "other": sorted(map(_box, read["other"]))}

    differences = _differences(text(golden), text(out))
    assert not differences, "\n".join(differences)
