from __future__ import annotations

from typing import Iterable

import numpy as np


def select_f1_threshold(targets: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(targets), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.size == 0 or y.size != p.size or np.unique(y).size < 2:
        raise ValueError("threshold selection requires aligned validation samples from both labels")
    # Classification changes only between adjacent distinct probabilities.
    # Evaluate those gap midpoints instead of observed sample scores.  Using an
    # observed positive score as the threshold puts the boundary directly on a
    # validation sample and is unnecessarily brittle under small score shifts
    # on another turbine.
    unique = np.unique(p)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], midpoints, [1.0])))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        prediction = p >= threshold
        tp = int(np.sum(prediction & (y == 1)))
        fp = int(np.sum(prediction & (y == 0)))
        fn = int(np.sum(~prediction & (y == 1)))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


def binary_metrics(targets: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, object]:
    y = np.asarray(list(targets), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.size == 0 or y.size != p.size or np.unique(y).size < 2:
        raise ValueError("binary metrics require aligned samples from both labels")
    prediction = p >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(y.size, 1)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall_range_label": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, np.finfo(float).eps)),
        "pr_auc_range_label": float(_average_precision(y, p)),
        "roc_auc_range_label": float(_roc_auc(y, p)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    positives = int(np.sum(y == 1))
    true_positives = np.cumsum(y[order] == 1)
    precision = true_positives / np.arange(1, y.size + 1)
    return float(np.sum(precision * (y[order] == 1)) / positives)


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    # Mann-Whitney formulation with average ranks for probability ties.
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_ranks = float(np.sum(ranks[y == 1]))
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)
