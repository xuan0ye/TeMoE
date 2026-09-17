from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def msa_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, float]:
    """Standard MMSA regression metrics (MOSI/MOSEI/SIMS convention).

    Reports MAE, Pearson correlation, 7-class and 5-class accuracy on the
    rounded score, binary accuracy / F1 in two conventions:
    - has0: 3-way boundary at 0 collapsed to non-negative vs negative (includes 0)
    - non0: neutral (==0) samples removed, strictly positive vs negative
    """
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)

    low, high = label_range
    mae = float(np.mean(np.abs(predictions - targets)))
    corr = _safe_corr(predictions, targets)

    span = int(round(high - low))
    pred_c = np.clip(np.round(predictions), low, high)
    true_c = np.clip(np.round(targets), low, high)
    acc7 = float(accuracy_score(true_c, pred_c))

    pred_5 = np.clip(np.round(predictions), -2, 2)
    true_5 = np.clip(np.round(targets), -2, 2)
    acc5 = float(accuracy_score(true_5, pred_5))

    acc2_has0, f1_has0 = _binary(predictions >= 0, targets >= 0)

    non_zero = targets != 0
    if non_zero.sum() > 0:
        acc2_non0, f1_non0 = _binary(predictions[non_zero] > 0, targets[non_zero] > 0)
    else:
        acc2_non0, f1_non0 = float("nan"), float("nan")

    return {
        "mae": mae,
        "corr": corr,
        "acc7": acc7,
        "acc5": acc5,
        "acc2_has0": acc2_has0,
        "f1_has0": f1_has0,
        "acc2_non0": acc2_non0,
        "f1_non0": f1_non0,
        "label_span": span,
    }


def _binary(pred_bool: np.ndarray, true_bool: np.ndarray) -> tuple[float, float]:
    pred_int = pred_bool.astype(int)
    true_int = true_bool.astype(int)
    acc = float(accuracy_score(true_int, pred_int))
    f1 = float(f1_score(true_int, pred_int, average="weighted", zero_division=0))
    return acc, f1


def _safe_corr(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.std() < 1e-8 or true.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])
