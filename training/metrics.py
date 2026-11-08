"""eDOCr2's evaluation helpers from its ``train_tools``, unchanged: character error rate and box IoU."""

from __future__ import annotations

import typing
from collections import Counter


# From train_tools.py (f6f9651).
def compare_characters(label, prediction):
    # Count occurrences of each character in label and prediction
    label_chars = Counter(label)    # e.g., {'1': 1, '4': 1, '0': 1}
    pred_chars = Counter(prediction)  # e.g., {'4': 1, '0': 1}

    correct_count = 0

    # Iterate over characters in the prediction
    for char in pred_chars:
        if char in label_chars:
            # Add the minimum of occurrences in both to correct_count
            correct_count += min(pred_chars[char], label_chars[char])
    return correct_count


# From train_tools.py (f6f9651).
def get_cer(
    preds: typing.Union[str, typing.List[str]],
    target: typing.Union[str, typing.List[str]],
    ) -> float:
    
    def edit_distance(prediction_tokens: typing.List[str], reference_tokens: typing.List[str]) -> int:
        """ Standard dynamic programming algorithm to compute the Levenshtein Edit Distance Algorithm

        Args:
            prediction_tokens: A tokenized predicted sentence
            reference_tokens: A tokenized reference sentence
        Returns:
            Edit distance between the predicted sentence and the reference sentence
        """
        # Initialize a matrix to store the edit distances
        dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]

        # Fill the first row and column with the number of insertions needed
        for i in range(len(prediction_tokens) + 1):
            dp[i][0] = i
        
        for j in range(len(reference_tokens) + 1):
            dp[0][j] = j

        # Iterate through the prediction and reference tokens
        for i, p_tok in enumerate(prediction_tokens):
            for j, r_tok in enumerate(reference_tokens):
                # If the tokens are the same, the edit distance is the same as the previous entry
                if p_tok == r_tok:
                    dp[i+1][j+1] = dp[i][j]
                # If the tokens are different, the edit distance is the minimum of the previous entries plus 1
                else:
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1

        # Return the final entry in the matrix as the edit distance     
        return dp[-1][-1]
    """ Update the cer score with the current set of references and predictions.

    Args:
        preds (typing.Union[str, typing.List[str]]): list of predicted sentences
        target (typing.Union[str, typing.List[str]]): list of target words

    Returns:
        Character error rate score
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer


# From train_tools.py (f6f9651).
def calculate_iou(predicted_polygon, ground_truth_polygon):
    """
    Calculate IoU (Intersection over Union) between two polygons.
    """
    from shapely.geometry import Polygon
    pred_poly = Polygon(predicted_polygon)
    gt_poly = Polygon(ground_truth_polygon)

    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0

    # Calculate intersection and union areas
    intersection_area = pred_poly.intersection(gt_poly).area
    union_area = pred_poly.union(gt_poly).area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou
