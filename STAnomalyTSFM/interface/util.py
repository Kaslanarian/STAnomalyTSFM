import numpy as np


def predict_by_score(
    score: np.ndarray,
    contamination: float,
    return_threshold: bool = False,
):
    pred = np.zeros_like(score)
    threshold = np.percentile(score, 1 - contamination)
    pred[score > threshold] = 1
    if return_threshold:
        return pred, threshold
    return
