"""The two networks: NAVER's CRAFT text detector and eDOCr2's CRNN recognizer."""

from .crnn import CRNN, load_crnn

__all__ = ["CRNN", "load_crnn"]
