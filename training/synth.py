"""Synthetic training data for eDOCr2's recognizers, without TensorFlow.

The image generator and its helpers are keras-ocr's ``data_generation`` and ``tools``, and the
detector's target maps keras-ocr's ``detection``, as eDOCr2 ships them; the balanced text generator,
the sample filter and the drawing-background compositing are eDOCr2's ``train_tools``. They are
copied unchanged apart from the imports and the backgrounds' directory, so that the data matches
what eDOCr2 trained on.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import typing
from collections import Counter

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from scipy import spatial
from shapely import geometry

from edocr2.ocr_utils import fit, read

# From data_generation.py (f6f9651).
LIGATURES = {"\U0000FB01": "fi", "\U0000FB02": "fl"}
LIGATURE_STRING = "".join(LIGATURES.keys())


# From data_generation.py (f6f9651).
def get_rotation_matrix(width, height, thetaX=0, thetaY=0, thetaZ=0):
    """Provide a rotation matrix about the center of a rectangle with
    a given width and height.

    Args:
        width: The width of the rectangle
        height: The height of the rectangle
        thetaX: Rotation about the X axis
        thetaY: Rotation about the Y axis
        thetaZ: Rotation about the Z axis

    Returns:
        A 3x3 transformation matrix
    """
    translate1 = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])
    rotX = np.array(
        [
            [1, 0, 0],
            [0, np.cos(thetaX), -np.sin(thetaX)],
            [0, np.sin(thetaX), np.cos(thetaX)],
        ]
    )
    rotY = np.array(
        [
            [np.cos(thetaY), 0, np.sin(thetaY)],
            [0, 1, 0],
            [-np.sin(thetaY), 0, np.cos(thetaY)],
        ]
    )
    rotZ = np.array(
        [
            [np.cos(thetaZ), -np.sin(thetaZ), 0],
            [np.sin(thetaZ), np.cos(thetaZ), 0],
            [0, 0, 1],
        ]
    )
    translate2 = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    M = translate1.dot(rotX).dot(rotY).dot(rotZ).dot(translate2)
    return M


# From data_generation.py (f6f9651).
def get_maximum_uniform_contour(image, fontsize, margin=0):
    """Get the largest possible contour of light or
    dark area in an image.

    Args:
        image: The image in which to find a contiguous area.
        fontsize: The fontsize for text. Will be used for blurring
            and for determining useful areas.
        margin: The minimum margin required around the image.

    Returns:
        A (contour, isDark) tuple. If no contour is found, both
        entries will be None.
    """
    if margin > 0:
        image = image[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(src=gray, ksize=(fontsize // 2, fontsize // 2))
    _, threshold = cv2.threshold(
        src=blurred, thresh=255 / 2, maxval=255, type=cv2.THRESH_BINARY
    )
    contoursDark = cv2.findContours(
        255 - threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    contoursLight = cv2.findContours(
        threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    areasDark = list(map(cv2.contourArea, contoursDark))
    areasLight = list(map(cv2.contourArea, contoursLight))
    maxDarkArea = max(areasDark) if areasDark else 0
    maxLightArea = max(areasLight) if areasLight else 0

    if max(maxDarkArea, maxLightArea) < (4 * fontsize) ** 2:
        return None, None

    contour = None
    isDark = None
    if areasDark and (not areasLight or maxDarkArea >= maxLightArea):
        contour = contoursDark[np.argmax(areasDark)]
        isDark = True
    else:
        contour = contoursLight[np.argmax(areasLight)]
        isDark = False
    if contour is not None:
        contour += margin
    return contour, isDark


# From data_generation.py (f6f9651).
def _strip_line(line):
    """Modify a line so that spaces are excluded."""
    first_character_index = next(
        (
            index
            for index, (box, character) in enumerate(line)
            if not character.isspace()
        ),
        None,
    )
    if first_character_index is None:
        return []
    last_character_index = len(line) - next(
        index
        for index, (box, character) in enumerate(reversed(line))
        if not character.isspace()
    )
    return line[first_character_index:last_character_index]


# From data_generation.py (f6f9651).
def _strip_lines(lines):
    """Modify a set of lines so that spaces are excluded."""
    lines = [line for line in lines if len(line) > 0]
    lines = [_strip_line(line) for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines


# From data_generation.py (f6f9651).
def convert_image_generator_to_recognizer_input(
    image_generator, max_string_length, target_width, target_height, margin=0
):
    """Convert an image generator created by get_image_generator
    to (image, sentence) tuples for training a recognizer.

    Args:
        image_generator: An image generator created by get_image_generator
        max_string_length: The maximum string length to allow
        target_width: The width to warp lines into
        target_height: The height to warp lines into
        margin: The margin to apply around a single line.
    """
    while True:
        image, lines = next(image_generator)
        if len(lines) == 0:
            continue
        for line in lines:
            line = _strip_line(line[:max_string_length])
            if not line:
                continue
            box, sentence = combine_line(line)

            # remove multiple sequential spaces
            while "  " in sentence:
                sentence = sentence.replace("  ", " ")

            crop = warpBox(
                image=image,
                box=box,
                target_width=target_width,
                target_height=target_height,
                margin=margin,
                skip_rotate=True,
            )
            yield crop, sentence


# From data_generation.py (f6f9651).
def draw_text_image(
    text,
    fontsize,
    height,
    width,
    fonts,
    use_ligatures=False,
    thetaX=0,
    thetaY=0,
    thetaZ=0,
    color=(0, 0, 0),
    permitted_contour=None,
    draw_contour=False,
):
    """Get a transparent image containing text.

    Args:
        text: The text to draw on the image
        fontsize: The size of text to show.
        height: The height of the output image
        width: The width of the output image
        fonts: A dictionary of {subalphabet: paths_to_font}
        thetaX: Rotation about the X axis
        thetaY: Rotation about the Y axis
        thetaZ: Rotation about the Z axis
        color: The color of drawn text
        permitted_contour: A contour defining which part of the image
            we can put text. If None, the entire canvas is permitted
            for text.
        use_ligatures: Whether to render ligatures. If True,
            ligatures are always used (with an initial check for support
            which sometimes yields false positives). If False, ligatures
            are never used.

    Returns:
        An (image, lines) tuple where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    if not use_ligatures:
        fonts = {
            subalphabet: PIL.ImageFont.truetype(font_path, size=fontsize)
            if font_path is not None
            else PIL.ImageFont.load_default()
            for subalphabet, font_path in fonts.items()
        }
    if use_ligatures:
        for subalphabet, font_path in fonts.items():
            ligatures_supported = True
            font = (
                PIL.ImageFont.truetype(font_path, size=fontsize)
                if font_path is not None
                else PIL.ImageFont.load_default()
            )
            for ligature in LIGATURES:
                try:
                    font.getsize(ligature)
                except UnicodeEncodeError:
                    ligatures_supported = False
                    break
            if ligatures_supported:
                del fonts[subalphabet]
                subalphabet += LIGATURE_STRING
            fonts[subalphabet] = font
        for insert, search in LIGATURES.items():
            for subalphabet in fonts.keys()():
                if insert in subalphabet:
                    text = text.replace(search, insert)
    character_font_pairs = [
        (
            character,
            next(
                font for subalphabet, font in fonts.items() if character in subalphabet
            ),
        )
        for character in text
    ]
    M = get_rotation_matrix(
        width=width, height=height, thetaZ=thetaZ, thetaX=thetaX, thetaY=thetaY
    )
    if permitted_contour is None:
        permitted_contour = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).astype("float32")
    character_sizes = np.array(
        [font.font.getsize(character) for character, font in character_font_pairs]
    )
    min_character_size = character_sizes.sum(axis=1).min()
    transformed_contour = compute_transformed_contour(
        width=width,
        height=height,
        fontsize=max(min_character_size, 1),
        M=M,
        contour=permitted_contour,
    )
    start_x = transformed_contour[:, 0].min()
    start_y = transformed_contour[:, 1].min()
    end_x = transformed_contour[:, 0].max()
    end_y = transformed_contour[:, 1].max()
    image = PIL.Image.new(mode="RGBA", size=(width, height), color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image)
    lines_raw: typing.List[typing.List[typing.Tuple[np.ndarray, str]]] = [[]]
    x = start_x
    y = start_y
    max_y = start_y
    out_of_space = False
    for character_index, (character, font) in enumerate(character_font_pairs):
        if out_of_space:
            break
        (character_width, character_height), (offset_x, offset_y) = character_sizes[
            character_index
        ]
        if character in LIGATURES:
            subcharacters = LIGATURES[character]
            dx = character_width / len(subcharacters)
        else:
            subcharacters = character
            dx = character_width
        x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        while not all(
            cv2.pointPolygonTest(contour=transformed_contour, pt=tuple([int(pt[0]),int(pt[1])]), measureDist=False)
            >= 0
            for pt in [(x, y), (x2, y), (x2, y2), (x, y2)]
        ):
            if x2 > end_x:
                dy = max(1, max_y - y)
                if y + dy > end_y:
                    out_of_space = True
                    break
                y += dy
                x = start_x
            else:
                x += fontsize
            if len(lines_raw[-1]) > 0:
                # We add a new line whether we have advanced
                # in the y-direction or not because we also want to separate
                # horizontal segments of text.
                lines_raw.append([])
            x2, y2 = (x + character_width + offset_x, y + character_height + offset_y)
        if out_of_space:
            break
        max_y = max(y + character_height + offset_y, max_y)
        draw.text(xy=(x, y), text=character, fill=color + (255,), font=font)
        for subcharacter in subcharacters:
            lines_raw[-1].append(
                (
                    np.array(
                        [
                            [x + offset_x, y + offset_y],
                            [x + dx + offset_x, y + offset_y],
                            [x + dx + offset_x, y2],
                            [x + offset_x, y2],
                        ]
                    ).astype("float32"),
                    subcharacter,
                )
            )
            x += dx
    image = cv2.warpPerspective(src=np.array(image), M=M, dsize=(width, height))
    if draw_contour:
        image = cv2.drawContours(
            image,
            contours=[permitted_contour.reshape((-1, 1, 2)).astype("int32")],
            contourIdx=0,
            color=(255, 0, 0, 255),
            thickness=int(width / 100),
        )
    lines_stripped = _strip_lines(lines_raw)
    lines_transformed = [
        [
            (cv2.perspectiveTransform(src=coords[np.newaxis], m=M)[0], character)
            for coords, character in line
        ]
        for line in lines_stripped
    ]
    return image, lines_transformed


# From data_generation.py (f6f9651).
def compute_transformed_contour(width, height, fontsize, M, contour, minarea=0.5):
    """Compute the permitted drawing contour
    on a padded canvas for an image of a given size.
    We assume the canvas is padded with one full image width
    and height on left and right, top and bottom respectively.

    Args:
        width: Width of image
        height: Height of image
        fontsize: Size of characters
        M: The transformation matrix
        contour: The contour to which we are limited inside
            the rectangle of size width / height
        minarea: The minimum area required for a character
            slot to qualify as being visible, expressed as
            a fraction of the untransformed fontsize x fontsize
            slot.
    """
    spacing = math.ceil(fontsize / 2)
    xslots = int(np.floor(width / spacing))
    yslots = int(np.floor(height / spacing))
    ys, xs = np.mgrid[:yslots, :xslots]
    basis = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape(
        (-1, 2)
    )
    basis *= spacing
    slots_pretransform = np.concatenate(
        [
            (basis + offset)[:, np.newaxis, :]
            for offset in [[0, 0], [spacing, 0], [spacing, spacing], [0, spacing]]
        ],
        axis=1,
    )
    slots = cv2.perspectiveTransform(
        src=slots_pretransform.reshape((1, -1, 2)).astype("float32"), m=M
    )[0]
    inside = (
        np.array(
            [
                cv2.pointPolygonTest(contour=contour, pt=tuple([int(x), int(y)]), measureDist=False) >= 0
                for x, y in slots
            ]
        )
        .reshape(-1, 4)
        .all(axis=1)
    )
    slots = slots.reshape(-1, 4, 2)
    areas = (
        np.abs(
            (slots[:, 0, 0] * slots[:, 1, 1] - slots[:, 0, 1] * slots[:, 1, 0])
            + (slots[:, 1, 0] * slots[:, 2, 1] - slots[:, 1, 1] * slots[:, 2, 0])
            + (slots[:, 2, 0] * slots[:, 3, 1] - slots[:, 2, 1] * slots[:, 3, 0])
            + (slots[:, 3, 0] * slots[:, 0, 1] - slots[:, 3, 1] * slots[:, 0, 0])
        )
        / 2
    )
    slots_filtered = slots_pretransform[(areas > minarea * spacing * spacing) & inside]
    temporary_image = cv2.drawContours(
        image=np.zeros((height, width), dtype="uint8"),
        contours=slots_filtered,
        contourIdx=-1,
        color=255,
    )
    temporary_image = cv2.dilate(
        src=temporary_image, kernel=np.ones((spacing, spacing))
    )
    newContours, _ = cv2.findContours(
        temporary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    x, y = slots_filtered[0][0]
    #print(x,y)
    contour = newContours[
        next(
            index
            for index, contour in enumerate(newContours)
            if cv2.pointPolygonTest(contour, tuple([int(x), int(y)]), False) >= 0
        )
    ][:, 0, :]
    return contour


# From data_generation.py (f6f9651).
def get_image_generator(
    height,
    width,
    font_groups,
    text_generator,
    font_size: typing.Union[int, typing.Tuple[int, int]] = 18,
    backgrounds: typing.List[typing.Union[str, np.ndarray]] = None,
    background_crop_mode="crop",
    rotationX: typing.Union[int, typing.Tuple[int, int]] = 0,
    rotationY: typing.Union[int, typing.Tuple[int, int]] = 0,
    rotationZ: typing.Union[int, typing.Tuple[int, int]] = 0,
    margin=0,
    use_ligatures=False,
    augmenter=None,
    draw_contour=False,
    draw_contour_text=False,
):
    """Create a generator for images containing text.

    Args:
        height: The height of the generated image
        width: The width of the generated image.
        font_groups: A dict mapping of { subalphabet: [path_to_font1, path_to_font2] }.
        text_generator: See get_text_generator
        font_size: The font size to use. Alternative, supply a tuple
            and the font size will be randomly selected between
            the two values.
        backgrounds: A list of paths to image backgrounds or actual images
            as numpy arrays with channels in RGB order.
        background_crop_mode: One of letterbox or crop, indicates
            how backgrounds will be resized to fit on the canvas.
        rotationX: The X-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        rotationY: The Y-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        rotationZ: The Z-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        margin: The minimum margin around the edge of the image.
        use_ligatures: Whether to render ligatures (see `draw_text_image`)
        augmenter: An image augmenter to be applied to backgrounds
        draw_contour: Draw the permitted contour onto images (debugging only)
        draw_contour_text: Draw the permitted contour inside the text
            drawing function.

    Yields:
        Tuples of (image, lines) where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    if backgrounds is None:
        backgrounds = [np.zeros((height, width, 3), dtype="uint8")]
    alphabet = "".join(font_groups.keys())
    #assert len(set(alphabet)) == len(
    #    alphabet
    #), "Each character can appear in the subalphabet for only one font group."
    for text, background_index, current_font_groups in zip(
        text_generator,
        itertools.cycle(range(len(backgrounds))),
        zip(
            *[
                itertools.cycle(
                    [
                        (subalphabet, font_filepath)
                        for font_filepath in font_group_filepaths
                    ]
                )
                for subalphabet, font_group_filepaths in font_groups.items()
            ]
        ),
    ):
        if background_index == 0:
            random.shuffle(backgrounds)
        current_font_groups = dict(current_font_groups)
        current_font_size = (
            np.random.randint(low=font_size[0], high=font_size[1])
            if isinstance(font_size, tuple)
            else font_size
        )
        current_rotation_X, current_rotation_Y, current_rotation_Z = [
            (
                np.random.uniform(low=rotation[0], high=rotation[1])
                if isinstance(rotation, tuple)
                else rotation
            )
            * np.pi
            / 180
            for rotation in [rotationX, rotationY, rotationZ]
        ]
        current_background_filepath_or_array = backgrounds[background_index]
        current_background = (
            read(current_background_filepath_or_array)
            if isinstance(current_background_filepath_or_array, str)
            else current_background_filepath_or_array
        )
        if augmenter is not None:
            current_background = augmenter(images=[current_background])[0]
        if (
            current_background.shape[0] != height
            or current_background.shape[1] != width
        ):
            current_background = fit(
                current_background,
                width=width,
                height=height,
                mode=background_crop_mode,
            )
        permitted_contour, isDark = get_maximum_uniform_contour(
            image=current_background, fontsize=current_font_size, margin=margin
        )
        if permitted_contour is None:
            # We can't draw on this background. Boo!
            continue
        random_color_values = np.random.randint(low=0, high=50, size=3)
        text_color = (
            tuple(np.array([255, 255, 255]) - random_color_values)
            if isDark
            else tuple(random_color_values)
        )
        text_image, lines = draw_text_image(
            text=text,
            width=width,
            height=height,
            fontsize=current_font_size,
            fonts=current_font_groups,
            thetaX=current_rotation_X,
            thetaY=current_rotation_Y,
            thetaZ=current_rotation_Z,
            use_ligatures=use_ligatures,
            permitted_contour=permitted_contour,
            color=text_color,
            draw_contour=draw_contour_text,
        )
        alpha = text_image[..., -1:].astype("float32") / 255
        image = (alpha * text_image[..., :3] + (1 - alpha) * current_background).astype(
            "uint8"
        )
        if draw_contour:
            image = cv2.drawContours(
                image,
                contours=[permitted_contour.reshape((-1, 1, 2)).astype("int32")],
                contourIdx=0,
                color=(255, 0, 0),
                thickness=int(width / 100),
            )
        yield image, lines


# From tools.py (f6f9651).
def get_rotated_width_height(box):
    """
    Returns the width and height of a rotated rectangle

    Args:
        box: A list of four points starting in the top left
        corner and moving clockwise.
    """
    w = (
        spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")
    ) / 2
    h = (
        spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")
    ) / 2
    return int(w[0][0]), int(h[0][0])


# From tools.py (f6f9651).
def get_rotated_box(
    points,
) -> typing.Tuple[np.ndarray, float,]:
    """Obtain the parameters of a rotated box.

    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[
            :-1
        ]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation


# From tools.py (f6f9651).
def warpBox(
    image,
    box,
    target_height=None,
    target_width=None,
    margin=0,
    cval=None,
    return_transform=False,
    skip_rotate=False,
):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.

    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """
    if cval is None:
        cval = (0, 0, 0) if len(image.shape) == 3 else 0
    if not skip_rotate:
        box, _ = get_rotated_box(box)
    w, h = get_rotated_width_height(box)
    assert (target_width is None and target_height is None) or (
        target_width is not None and target_height is not None
    ), "Either both or neither of target width and height must be provided."
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(
        src=box,
        dst=np.array(
            [
                [margin, margin],
                [scale * w - margin, margin],
                [scale * w - margin, scale * h - margin],
                [margin, scale * h - margin],
            ]
        ).astype("float32"),
    )
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    target_shape = (
        (target_height, target_width, 3)
        if len(image.shape) == 3
        else (target_height, target_width)
    )
    full = (np.zeros(target_shape) + cval).astype("uint8")
    full[: crop.shape[0], : crop.shape[1]] = crop
    if return_transform:
        return full, M
    return full


# From tools.py (f6f9651).
def combine_line(line):
    """Combine a set of boxes in a line into a single bounding
    box.

    Args:
        line: A list of (box, character) entries

    Returns:
        A (box, text) tuple
    """
    text = "".join(
        [character if character is not None else "" for _, character in line]
    )
    box = np.concatenate(
        [coords[:2] for coords, _ in line]
        + [np.array([coords[3], coords[2]]) for coords, _ in reversed(line)]
    ).astype("float32")
    first_point = box[0]
    rectangle = cv2.minAreaRect(box)
    box = cv2.boxPoints(rectangle)

    # Put the points in clockwise order
    box = np.array(np.roll(box, -np.linalg.norm(box - first_point, axis=1).argmin(), 0))
    return box, text


# From tools.py (f6f9651).
def fix_line(line):
    """Given a list of (box, character) tuples, return a revised
    line with a consistent ordering of left-to-right or top-to-bottom,
    with each box provided with (top-left, top-right, bottom-right, bottom-left)
    ordering.

    Returns:
        A tuple that is the fixed line as well as a string indicating
        whether the line is horizontal or vertical.
    """
    line = [(get_rotated_box(box)[0], character) for box, character in line]
    centers = np.array([box.mean(axis=0) for box, _ in line])
    sortedx = centers[:, 0].argsort()
    sortedy = centers[:, 1].argsort()
    if np.diff(centers[sortedy][:, 1]).sum() > np.diff(centers[sortedx][:, 0]).sum():
        return [line[idx] for idx in sortedy], "vertical"
    return [line[idx] for idx in sortedx], "horizontal"


# From detection.py (f6f9651).
def get_gaussian_heatmap(size=512, distanceRatio=3.34):
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    x, y = np.meshgrid(v, v)
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype("uint8")


# From detection.py (f6f9651).
def compute_maps(heatmap, image_height, image_width, lines):
    assert image_height % 2 == 0, "Height must be an even number"
    assert image_width % 2 == 0, "Width must be an even number"

    textmap = np.zeros((image_height // 2, image_width // 2)).astype("float32")
    linkmap = np.zeros((image_height // 2, image_width // 2)).astype("float32")

    src = np.array(
        [
            [0, 0],
            [heatmap.shape[1], 0],
            [heatmap.shape[1], heatmap.shape[0]],
            [0, heatmap.shape[0]],
        ]
    ).astype("float32")

    for line in lines:
        line, orientation = fix_line(line)
        previous_link_points = None
        for [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], c in line:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                lambda v: max(v, 0), [x1, y1, x2, y2, x3, y3, x4, y4]
            )
            if c == " ":
                previous_link_points = None
                continue
            yc = (y4 + y1 + y3 + y2) / 4
            xc = (x1 + x2 + x3 + x4) / 4
            if orientation == "horizontal":
                current_link_points = (
                    np.array(
                        [
                            [(xc + (x1 + x2) / 2) / 2, (yc + (y1 + y2) / 2) / 2],
                            [(xc + (x3 + x4) / 2) / 2, (yc + (y3 + y4) / 2) / 2],
                        ]
                    )
                    / 2
                )
            else:
                current_link_points = (
                    np.array(
                        [
                            [(xc + (x1 + x4) / 2) / 2, (yc + (y1 + y4) / 2) / 2],
                            [(xc + (x2 + x3) / 2) / 2, (yc + (y2 + y3) / 2) / 2],
                        ]
                    )
                    / 2
                )
            character_points = (
                np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype("float32") / 2
            )
            # pylint: disable=unsubscriptable-object
            if previous_link_points is not None:
                if orientation == "horizontal":
                    link_points = np.array(
                        [
                            previous_link_points[0],
                            current_link_points[0],
                            current_link_points[1],
                            previous_link_points[1],
                        ]
                    )
                else:
                    link_points = np.array(
                        [
                            previous_link_points[0],
                            previous_link_points[1],
                            current_link_points[1],
                            current_link_points[0],
                        ]
                    )
                ML = cv2.getPerspectiveTransform(
                    src=src,
                    dst=link_points.astype("float32"),
                )
                linkmap += cv2.warpPerspective(
                    heatmap, ML, dsize=(linkmap.shape[1], linkmap.shape[0])
                ).astype("float32")
            MA = cv2.getPerspectiveTransform(
                src=src,
                dst=character_points,
            )
            textmap += cv2.warpPerspective(
                heatmap, MA, dsize=(textmap.shape[1], textmap.shape[0])
            ).astype("float32")
            # pylint: enable=unsubscriptable-object
            previous_link_points = current_link_points
    return (
        np.concatenate(
            [textmap[..., np.newaxis], linkmap[..., np.newaxis]], axis=2
        ).clip(0, 255)
        / 255
    )


# From train_tools.py (f6f9651).
def get_balanced_text_generator(alphabet, string_length=(5, 10), lowercase=False, bias_chars = '', bias_factor = 0.3):
    '''
    Generates batches of sentences ensuring perfectly balanced symbol distribution.
    Args:
        alphabet: string of characters
        batch_size: number of sentences per batch
        string_length: tuple defining range of sentence length
        lowercase: convert alphabet to lowercase
    Return:
        list of sentence strings
    '''
    # Initialize a counter to track the number of times each character is used
    symbol_counter = Counter({char: 0 for char in alphabet})
    
    while True:
        # Calculate the total number of generated symbols
        total_generated = sum(symbol_counter.values())

        # Adjust probabilities to balance the frequency of each symbol
        weights = {}
        for char in alphabet:
            # Apply the bias factor for specified characters
            weight = total_generated - symbol_counter[char] + 1
            if char in bias_chars:
                weight += bias_factor
            weights[char] = weight
        total_weight = sum(weights.values())
        probabilities = [weights[char] / total_weight for char in alphabet]

        # Sample a sentence based on the adjusted probabilities
        sentence = random.choices(alphabet, weights=probabilities, k=random.randint(string_length[0], string_length[1]))
        sentence = "".join(sentence)

        # Update the symbol counter
        symbol_counter.update(sentence)

        if lowercase:
            sentence = sentence.lower()
        
        yield sentence


# From train_tools.py (f6f9651).
def get_backgrounds(height, width, samples):
    backgrounds = []
    backg_path = os.path.join(os.path.dirname(__file__), 'backgrounds')
    backg_files = os.listdir(backg_path)
    for _ in range(samples):
        backg_file = random.choice(backg_files)
        img = cv2.imread(os.path.join(backg_path, backg_file))
        y, x = random.randint(0, img.shape[0] - height), random.randint(0, img.shape[1] - width)
        backg = img[y : y + height, x : x + width][:]
        backgrounds.append(backg)

    return backgrounds


# From train_tools.py (f6f9651).
def filter_wrong_samples(generator, white_pixel_threshold=0.05):
    """A generator wrapper that filters out samples with too many white pixels.
    
    Args:
    generator: The original generator that produces image samples.
    white_pixel_threshold: The maximum allowed ratio of white pixels.
    
    Yields:
    Valid samples that meet the white pixel threshold criteria.
    """
    for image, text in generator:
        # Convert image to grayscale to count white pixels
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create a binary image (white pixels = 255, other = 0)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Calculate total pixels and the number of white pixels
        total_pixels = binary_image.size
        white_pixels = np.sum(binary_image == 255)
        
        # Calculate the percentage of white pixels
        white_pixel_ratio = white_pixels / total_pixels
        
        # Yield the sample only if the white pixel ratio is within the acceptable threshold
        if white_pixel_ratio >= white_pixel_threshold:
            yield cv2.bitwise_not(image), text
        '''else:
            print(f"Skipping sample due to low white pixel ratio ({white_pixel_ratio:.2%})")'''


# From train_tools.py (f6f9651).
def generate_drawing_imgs(image_gen_params, backgrounds):
    def check_overlap(text_img, background_img):
            """Check if there is an overlap between black pixels of the background and white pixels of the text.
            Args:
            text_img: A binary image where the text is white (255) on black (0).
            background_img: A grayscale or RGB background image.
            Returns:
            bool: True if there is an overlap, False otherwise.
            """
            # Ensure both images are of the same size
            if text_img.shape != background_img.shape[:2]:
                raise ValueError("Text image and background image must have the same dimensions")
            
            # Convert background to grayscale if it's RGB
            if len(background_img.shape) == 3:
                background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
            else:
                background_gray = background_img

            # Identify where the text image has white pixels (text pixels)
            text_mask = text_img < 127
            # Identify where the background has black pixels (0 value)
            background_black_mask = background_gray < 127
            # Check if any black background pixels overlap with the white text pixels
            overlap = np.any(np.logical_and(text_mask, background_black_mask))

            return overlap

    def apply_text_on_background(text_img, text_binary, background_img):
        """Apply the text image over the background, assuming no overlap."""
        # Create a mask where text_binary is white (255), ndicating text
        text_mask = text_binary == 0
        
        # Create a copy of background_img to avoid modifying the original image
        result = background_img.copy()

        inverted_text_img = cv2.bitwise_not(text_img)
        result[text_mask] = inverted_text_img[text_mask]

        return result
    
    def compact_bounding_box(box_group):
        from edocr2.tools.ocr_pipelines import group_polygons_by_proximity
        box_groups = []
        for b in box_group:
            for xy, _ in b:
                box_groups.append(xy)
                    
        box_groups = group_polygons_by_proximity(box_groups, eps = 10)
        
        dummy_char = '1'
        dummy_box_groups = []

        for box in box_groups:
            dummy_box_groups.append([(np.array(box).astype(np.int32), dummy_char)])

        return dummy_box_groups
    
    def reposition(text_img, lines):
        new_lines = []
        for line in lines:
            x_coords = []
            y_coords = []
            for li in line:
                x_coords.extend([li[0][0][0], li[0][1][0], li[0][2][0], li[0][3][0]])  # [x1, x2, x3, x4]
                y_coords.extend([li[0][0][1], li[0][1][1], li[0][2][1], li[0][3][1]])  # [y1, y2, y3, y4]
            
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            
            # Crop the text region using the bounding box coordinates
            cropped_text = text_img[y_min:y_max, x_min:x_max]
            x_offset = random.randint(10, text_img.shape[1] - x_max + x_min - 10)
            y_offset = random.randint(10, text_img.shape[0] - y_max + y_min - 10)
            
            text_img[y_offset:y_offset+ cropped_text.shape[0], x_offset:x_offset+ cropped_text.shape[1]] = cropped_text
            text_img[y_min:y_max, x_min:x_max] = 0
            new_line = []
            for li in line:
                new_li = []
                for coord in li[0]:  # Iterate through each (x, y) pair in the bounding box
                    new_x = coord[0] - x_min + x_offset
                    new_y = coord[1] - y_min + y_offset
                    new_li.append([new_x, new_y])
                new_line.append([new_li, li[1]])
            new_lines.append(new_line)

        return text_img, new_lines
    

    while True:
        backg = random.choice(backgrounds)
        # Initialize the final image as the background
        image = backg.copy()
        lines = []  # Store the bounding boxes for all text images

        # Randomly choose a number of text images to place (between 1 and 5, for example)
        num_images = random.randint(1, 5)

        for _ in range(num_images):
            for _ in range(100):  # Retry mechanism if overlap occurs
                image_gen = get_image_generator(**image_gen_params)
                text_img, new_lines = next(image_gen)
                text_img, new_lines = reposition(text_img, new_lines)
                _, binary_text_img = cv2.threshold(cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)
                # Check if the new text image overlaps with the current image
                
                if not check_overlap(binary_text_img, image):
                    # If no overlap, apply the text image onto the background
                    image = apply_text_on_background(text_img, binary_text_img, image)

                    # Compact the bounding boxes and add them to the list
                    new_lines = compact_bounding_box(new_lines)
                    lines.extend(new_lines)
                    break  # Exit the loop once the image has been successfully placed
            else:
                continue  # Retry if overlap occurred

        # Yield the final image with the applied text and the compacted bounding boxes
        yield image, lines
