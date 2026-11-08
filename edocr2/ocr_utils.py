"""Image and box helpers the pipeline takes from keras-ocr.

Taken from keras-ocr (github.com/faustomorales/keras-ocr, MIT License, Fausto Morales) by way of
eDOCr2, with behaviour unchanged: the outputs of eDOCr2-pytorch must match eDOCr2's. Only reading an
image from a URL was dropped, with its ``validators`` dependency.
"""

from __future__ import annotations

import io
import os

import cv2
import numpy as np


def read(filepath_or_buffer: str | io.BytesIO | np.ndarray) -> np.ndarray:
    """An image as an array: a path or buffer is decoded and turned to RGB, an array is returned as is."""
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    else:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(f"Could not find image at path: {filepath_or_buffer}")
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_input(image: np.ndarray) -> np.ndarray:
    """CRAFT's input: the image normalized by ImageNet's mean and deviation, channel order kept."""
    image = image.astype("float32")
    image -= np.array([0.485, 0.456, 0.406]) * 255
    image /= np.array([0.229, 0.224, 0.225]) * 255
    return image


def get_boxes(y_pred, detection_threshold=0.7, text_threshold=0.4, link_threshold=0.4, size_threshold=10):
    """Word boxes, four corners each in input pixels, from CRAFT's text and link heatmaps.

    The boxes are keras-ocr's exactly, but each component is worked on inside its own window
    (its bounding box, the dilation margin and a zero border) rather than over the whole map,
    which keras-ocr and CRAFT redo once per component.
    """
    box_groups = []
    for y_pred_cur in y_pred:
        textmap = y_pred_cur[..., 0]
        linkmap = y_pred_cur[..., 1]
        img_h, img_w = textmap.shape
        _, text_score = cv2.threshold(textmap, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap, thresh=link_threshold, maxval=1, type=cv2.THRESH_BINARY)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.clip(text_score + link_score, 0, 1).astype("uint8"), connectivity=4
        )
        keep = ~np.logical_and(link_score, text_score)  # pixels both text and link are cut from every word
        boxes = []
        for component_id in np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= size_threshold):
            if component_id == 0:  # the background
                continue
            x, y, w, h, size = stats[component_id]
            # A component whose strongest text score is below the detection threshold is not a word.
            bounds = (slice(y, y + h), slice(x, x + w))
            if textmap[bounds][labels[bounds] == component_id].max() < detection_threshold:
                continue
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            # One pixel more where the map goes on, so findContours meets zeros where the map did.
            wx, wy = max(sx - 1, 0), max(sy - 1, 0)
            window = (slice(wy, min(ey + 1, img_h)), slice(wx, min(ex + 1, img_w)))
            segmap = np.where((labels[window] == component_id) & keep[window], 255, 0).astype("uint8")
            inner = (slice(sy - wy, ey - wy), slice(sx - wx, ex - wx))
            segmap[inner] = cv2.dilate(segmap[inner], cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))
            contours = cv2.findContours(segmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour = contours[0] + np.array([wx, wy], dtype=contours[0].dtype)  # back to map pixels
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:  # a diamond: take the axis-aligned bounds instead
                left, right = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                top, bottom = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
            else:  # clockwise, starting at the top-left corner
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)  # the heatmaps are at half the input's resolution
        box_groups.append(np.array(boxes))
    return box_groups


def adjust_boxes(boxes, scale=1, boxes_format="boxes"):
    """Boxes scaled by ``scale``, in keras-ocr's ``boxes``, ``lines`` or ``predictions`` format."""
    if scale == 1:
        return boxes
    if boxes_format == "boxes":
        boxes = np.array(boxes)
        # NumPy 1 multiplied float32 boxes by a float64 scalar in float32; NumPy 2 promotes to float64,
        # which moves the corners shapely starts its rectangles from (eDOCr2's outputs were made on 1).
        return boxes * (boxes.dtype.type(scale) if boxes.dtype == np.float32 else scale)
    if boxes_format == "lines":
        return [[(np.array(box) * scale, character) for box, character in line] for line in boxes]
    if boxes_format == "predictions":
        return [(word, np.array(box) * scale) for word, box in boxes]
    raise NotImplementedError(f"Unsupported boxes format: {boxes_format}")


def fit(image, width: int, height: int, cval: int = 255, mode="letterbox", return_scale=False):
    """The image resized to fit ``width`` x ``height``, letterboxed with ``cval`` or cropped.

    A letterboxed result always has three channels, a grayscale input included.
    """
    fitted = None
    x_scale = width / image.shape[1]
    y_scale = height / image.shape[0]
    if x_scale == 1 and y_scale == 1:
        fitted = image
        scale = 1
    elif (x_scale <= y_scale and mode == "letterbox") or (x_scale >= y_scale and mode == "crop"):
        scale = width / image.shape[1]
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == "letterbox":
            fitted = np.zeros((height, width, 3), dtype="uint8") + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[: image.shape[0], : image.shape[1]] = image[:height, :width]
        elif mode == "crop":
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
    if not return_scale:
        return fitted
    return fitted, scale


def read_and_fit(filepath_or_array: str | np.ndarray, width: int, height: int, cval: int = 255, mode="letterbox"):
    """``fit`` applied to an image path or array."""
    image = read(filepath_or_array) if isinstance(filepath_or_array, str) else filepath_or_array
    return fit(image=image, width=width, height=height, cval=cval, mode=mode)
