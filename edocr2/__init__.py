"""eDOCr2-pytorch: read dimensions, GD&T frames and title blocks on mechanical engineering drawings, in PyTorch.

A continuation of eDOCr2 (github.com/javvi51/edocr2) without TensorFlow. See NOTICE for where
each part comes from.
"""

from .detector import Detector
from .drawing import DrawingResult, Models, load_models, read_drawing
from .recognizer import Recognizer

__version__ = "0.1.0.dev0"

__all__ = ["Detector", "DrawingResult", "Models", "Recognizer", "load_models", "read_drawing"]
