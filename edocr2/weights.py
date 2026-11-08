"""The weights eDOCr2-pytorch reads, where each was first published, and fetching them.

eDOCr2-pytorch redistributes no weights. Each file is fetched from the release that published it and
checked against its sha256: NAVER's CRAFT detector from craft-detector's release, the file of
CRAFT's original download, and eDOCr2's two recognizers from eDOCr2's release, read straight from
their ``.keras`` files.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Weight:
    filename: str
    url: str
    sha256: str


_CRAFT = "https://github.com/wkpark/CRAFT-pytorch/releases/download/v0.0.2/"
_EDOCR2 = "https://github.com/javvi51/edocr2/releases/download/v1.0.0/"

# CRAFT's general model (clovaai, trained on SynthText, IC13 and IC17).
CRAFT_MLT_25K = Weight(
    "craft_mlt_25k.pth",
    _CRAFT + "craft_mlt_25k.pth",
    "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
)
DIMENSIONS = Weight(
    "recognizer_dimensions_2.keras",
    _EDOCR2 + "recognizer_dimensions_2.keras",
    "38cef806618da8b1d5c83c207fb5e4ee6964f7f14dc3f767a55bac96e9f54361",
)
DIMENSIONS_ALPHABET = Weight(
    "recognizer_dimensions_2.txt",
    _EDOCR2 + "recognizer_dimensions_2.txt",
    "d0d661bf06b9cdc7d4acc4b0fdce5be8d61c7c352c3e538efcf05a5a5e24634b",
)
GDTS = Weight(
    "recognizer_gdts.keras",
    _EDOCR2 + "recognizer_gdts.keras",
    "5996b84ffd1f1acabba6428056ceb6fe056a11c3b3d94eb9ee2c312414acbca7",
)
GDTS_ALPHABET = Weight(
    "recognizer_gdts.txt",
    _EDOCR2 + "recognizer_gdts.txt",
    "931160e26963d9199fdd4218709552a7b2959666416f57ca51dfd689c24880a4",
)
ALL = (CRAFT_MLT_25K, DIMENSIONS, DIMENSIONS_ALPHABET, GDTS, GDTS_ALPHABET)


def cache_dir() -> Path:
    """Where downloaded weights are kept: ``EDOCR2_CACHE_DIR``, else ``~/.cache/edocr2``."""
    return Path(os.environ.get("EDOCR2_CACHE_DIR") or Path.home() / ".cache" / "edocr2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(weight: Weight, directory: str | Path | None = None) -> Path:
    """The local path of ``weight``, checked against its sha256.

    Given ``directory``, or ``EDOCR2_WEIGHTS_DIR``, the file must already be there and nothing is
    downloaded, as an offline install needs. Otherwise it is downloaded into `cache_dir` once.
    """
    given = directory or os.environ.get("EDOCR2_WEIGHTS_DIR")
    if given:
        path = Path(given) / weight.filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: download it from {weight.url}")
    else:
        path = cache_dir() / weight.filename
        if not path.is_file() or _sha256(path) != weight.sha256:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A file of its own per download: processes filling an empty cache at once never write
            # into one file, and each puts the same checked bytes in place.
            fd, partial = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
            os.close(fd)
            try:
                urllib.request.urlretrieve(weight.url, partial)
                if _sha256(Path(partial)) != weight.sha256:
                    raise ValueError(f"the download of {weight.url} does not match its sha256")
                os.replace(partial, path)
            finally:
                if os.path.exists(partial):
                    os.unlink(partial)
    if _sha256(path) != weight.sha256:
        raise ValueError(f"{path} does not match the sha256 of {weight.url}")
    return path
