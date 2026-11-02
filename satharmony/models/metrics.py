"""Segmentation metrics for the cloud / shadow baseline.

Pure NumPy so the metrics can be unit-tested and reused without torch.  Every
metric is derived from a single ``num_classes x num_classes`` confusion matrix
(rows = reference, columns = prediction), which is what the G4 report stores.

The class order is the CloudSEN12 one used everywhere in the package:
``0 = clear``, ``1 = thick cloud``, ``2 = thin cloud``, ``3 = cloud shadow``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Class names in CloudSEN12 order.
CLASS_NAMES: tuple[str, ...] = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")

#: Number of classes of the baseline.
NUM_CLASSES = len(CLASS_NAMES)


def confusion_matrix(
    pred: NDArray[np.integer],
    target: NDArray[np.integer],
    num_classes: int = NUM_CLASSES,
    ignore_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """Confusion matrix of a segmentation.

    Args:
        pred: Predicted class per pixel.
        target: Reference class per pixel.
        num_classes: Number of classes.
        ignore_mask: Optional boolean mask; ``True`` pixels are dropped.

    Returns:
        ``(num_classes, num_classes)`` int64 matrix where the row is the
        reference class and the column is the prediction.

    Raises:
        ValueError: If shapes differ or an out-of-range class appears.
    """
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} and target {target.shape} differ")
    if ignore_mask is not None:
        keep = ~np.asarray(ignore_mask, dtype=bool).ravel()
        pred, target = pred[keep], target[keep]
    if pred.size and (
        pred.min() < 0 or pred.max() >= num_classes or target.min() < 0 or target.max() >= num_classes
    ):
        raise ValueError(f"classes must be in [0, {num_classes - 1}]")
    index = target.astype(np.int64) * num_classes + pred.astype(np.int64)
    counts = np.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64)


def iou_per_class(matrix: NDArray[np.integer]) -> dict[str, float]:
    """Intersection over union per class.

    Args:
        matrix: Confusion matrix (reference rows, prediction columns).

    Returns:
        ``{class_name: iou}``; a class with an empty union returns ``nan``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - true_positive
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, true_positive / union, np.nan)
    return {name: float(iou[i]) for i, name in enumerate(CLASS_NAMES[: matrix.shape[0]])}


def mean_iou(matrix: NDArray[np.integer], ignore_absent: bool = True) -> float:
    """Mean IoU over the classes.

    Args:
        matrix: Confusion matrix.
        ignore_absent: If True (default) classes with no reference and no
            prediction pixels are skipped, which is the usual convention for
            a test split. If False they count as 0.

    Returns:
        The mean, or ``nan`` when there is nothing to average.
    """
    values = np.array(list(iou_per_class(matrix).values()), dtype=np.float64)
    values = (
        values[~np.isnan(values)] if ignore_absent else np.nan_to_num(values, nan=0.0)
    )
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def per_class_scores(matrix: NDArray[np.integer]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1, IoU and support per class.

    Args:
        matrix: Confusion matrix.

    Returns:
        ``{class_name: {"precision", "recall", "f1", "iou", "support"}}``;
        undefined ratios are ``nan`` and ``support`` is an int.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(matrix)
    reference = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    iou = np.array(list(iou_per_class(matrix).values()), dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(CLASS_NAMES[: matrix.shape[0]]):
        precision = tp[index] / predicted[index] if predicted[index] else np.nan
        recall = tp[index] / reference[index] if reference[index] else np.nan
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan
        out[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou[index]),
            "support": float(reference[index]),
        }
    return out


def overall_accuracy(matrix: NDArray[np.integer]) -> float:
    """Fraction of pixels on the diagonal."""
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else float("nan")


def cohen_kappa(matrix: NDArray[np.integer]) -> float:
    """Cohen's kappa of a confusion matrix.

    Args:
        matrix: Confusion matrix.

    Returns:
        Kappa, or ``nan`` when the expected agreement is 1 (single class).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total * total))
    if expected >= 1.0:
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def evaluate(
    pred: NDArray[np.integer],
    target: NDArray[np.integer],
    ignore_mask: NDArray[np.bool_] | None = None,
) -> dict:
    """Full metric bundle for one prediction.

    Args:
        pred: Predicted classes.
        target: Reference classes.
        ignore_mask: Pixels to drop.

    Returns:
        JSON-friendly dict with the confusion matrix (nested lists), per-class
        scores, mean IoU, accuracy and kappa.
    """
    matrix = confusion_matrix(pred, target, ignore_mask=ignore_mask)
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class_scores(matrix),
        "mean_iou": mean_iou(matrix),
        "mean_iou_all_classes": mean_iou(matrix, ignore_absent=False),
        "overall_accuracy": overall_accuracy(matrix),
        "cohen_kappa": cohen_kappa(matrix),
        "n_pixels": int(matrix.sum()),
    }


# ── Decision rule ─────────────────────────────────────────────────────────
def apply_class_bias(
    probabilities: NDArray[np.floating],
    bias: NDArray[np.floating] | list[float] | tuple[float, ...],
    num_classes: int = NUM_CLASSES,
) -> NDArray[np.uint8]:
    """Argmax over ``log(p) + bias``.

    A per-class bias is the cheapest way to move the decision rule away from
    the implicit prior of the training labels: a positive bias recalls more of
    that class, a negative one suppresses it.  ``bias = 0`` is plain argmax.

    Args:
        probabilities: ``(..., num_classes, H, W)`` probabilities or a single
            ``(num_classes, H, W)`` map.
        bias: Per-class bias of length ``num_classes``.
        num_classes: Number of classes.

    Returns:
        ``(..., H, W)`` uint8 classes.

    Raises:
        ValueError: If the bias length or the class axis does not match.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    if bias.size != num_classes:
        raise ValueError(f"bias has {bias.size} values, expected {num_classes}")
    if probabilities.shape[-3] != num_classes:
        raise ValueError(
            f"class axis has {probabilities.shape[-3]} values, expected {num_classes}"
        )
    logits = np.log(np.maximum(probabilities, 1e-8)) + bias.reshape(-1, 1, 1)
    return logits.argmax(axis=-3).astype(np.uint8)


def tune_class_bias(
    probabilities: NDArray[np.floating],
    target: NDArray[np.integer],
    num_classes: int = NUM_CLASSES,
    step: float = 0.5,
    rounds: int = 4,
    lower: float = -3.0,
    upper: float = 3.0,
) -> dict:
    """Coordinate ascent on the per-class bias that maximises mean IoU.

    The bias must be tuned on a split that is **not** the test split (the
    validation split of the fixture), otherwise the gain is not honest.  Each
    round visits the classes in order and moves the bias of one class over a
    grid while the others stay fixed; the step halves every round.

    Args:
        probabilities: ``(N, num_classes, H, W)`` predicted probabilities.
        target: ``(N, H, W)`` reference classes.
        num_classes: Number of classes.
        step: Initial grid step.
        rounds: Number of refinement rounds.
        lower: Lower bound of the bias.
        upper: Upper bound of the bias.

    Returns:
        Dict with ``bias`` (list of floats), ``mean_iou``, ``iou`` per class,
        ``baseline_mean_iou`` and ``confusion_matrix``.

    Raises:
        ValueError: If shapes or steps are invalid.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    target = np.asarray(target)
    if probabilities.ndim != 4 or probabilities.shape[1] != num_classes:
        raise ValueError("probabilities must be (N, num_classes, H, W)")
    if probabilities.shape[0] != target.shape[0]:
        raise ValueError("probabilities and target have different chip counts")
    if step <= 0 or rounds < 1:
        raise ValueError("step must be positive and rounds >= 1")

    log_probabilities = np.log(np.maximum(probabilities, 1e-8))
    flat_target = target.reshape(-1)

    def score(bias: np.ndarray) -> float:
        predicted = (log_probabilities + bias.reshape(1, -1, 1, 1)).argmax(axis=1)
        return mean_iou(confusion_matrix(predicted.reshape(-1), flat_target, num_classes=num_classes))

    bias = np.zeros(num_classes, dtype=np.float64)
    best = score(bias)
    baseline = best
    current_step = step
    for _ in range(rounds):
        for index in range(num_classes):
            candidates = np.arange(
                max(lower, bias[index] - current_step),
                min(upper, bias[index] + current_step) + 1e-9,
                current_step / 4.0,
            )
            local_best, local_bias = best, bias[index]
            for candidate in candidates:
                trial = bias.copy()
                trial[index] = candidate
                value = score(trial)
                if value > local_best + 1e-9:
                    local_best, local_bias = value, candidate
            bias[index] = local_bias
            best = local_best
        current_step /= 2.0

    predicted = apply_class_bias(probabilities, bias, num_classes=num_classes)
    matrix = confusion_matrix(predicted, flat_target, num_classes=num_classes)
    return {
        "bias": [float(value) for value in bias],
        "mean_iou": mean_iou(matrix),
        "baseline_mean_iou": float(baseline),
        "iou": iou_per_class(matrix),
        "confusion_matrix": matrix.tolist(),
    }
