"""Losses for the cloud / shadow segmentation baseline (G4b).

The G4 baseline used a weighted cross-entropy plus soft Dice.  Thin cloud was
the weak class (IoU 0.21 on the pseudo test), so G4b adds the two levers that
usually close that gap without touching the labels:

* **Focal modulation** ``(1 - p_t) ** gamma``, which stops the thousands of
  easy clear pixels from dominating the gradient.
* **Stronger class priors** (inverse frequency instead of inverse square root),
  which raise the recall of the 6 % thin-cloud pixels before the decision rule
  is tuned.

Everything is pure torch so it can be unit-tested on CPU in milliseconds.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def class_weights(
    counts: torch.Tensor | list[int] | tuple[int, ...],
    mode: str = "sqrt",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-class weights from the pixel counts of the training split.

    Args:
        counts: Pixels per class.
        mode: ``sqrt`` (inverse square root, the G4 baseline), ``inverse``
            (plain inverse frequency, much stronger on thin cloud) or ``none``
            (all ones).
        eps: Floor applied to the frequencies before inverting.

    Returns:
        float32 weights normalised to mean 1.

    Raises:
        ValueError: If ``mode`` is unknown or a count is negative.
    """
    weights = torch.as_tensor(counts, dtype=torch.float64).flatten()
    if torch.any(weights < 0):
        raise ValueError("counts must be non-negative")
    if mode == "none":
        return torch.ones_like(weights, dtype=torch.float32)
    if mode not in {"sqrt", "inverse"}:
        raise ValueError(f"unknown class-weight mode: {mode}")
    frequencies = weights / weights.sum().clamp_min(1.0)
    if mode == "sqrt":
        values = 1.0 / torch.sqrt(frequencies.clamp_min(eps))
    else:
        values = 1.0 / frequencies.clamp_min(eps)
    values = values / values.mean()
    return values.to(torch.float32)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-class soft Dice loss.

    Args:
        logits: ``(B, C, H, W)`` model output.
        target: ``(B, H, W)`` int64 classes.
        num_classes: Number of classes.
        smooth: Additive smoothing of the ratio.
        weight: Optional per-class weights of shape ``(C,)`` applied to the
            per-class Dice before averaging.

    Returns:
        Scalar loss.
    """
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    union = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    if weight is not None:
        weights = weight.to(dice.dtype).to(dice.device)
        return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(1e-6)
    return 1.0 - dice.mean()


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Class-weighted focal cross-entropy.

    ``alpha_t * (1 - p_t) ** gamma * CE``, with ``alpha_t`` the class weight of
    the reference class.  ``gamma = 0`` reduces to plain weighted
    cross-entropy.

    Args:
        logits: ``(B, C, H, W)`` model output.
        target: ``(B, H, W)`` int64 classes.
        gamma: Focusing parameter; 0 disables the modulation.
        weight: Optional per-class weights of shape ``(C,)``.
        label_smoothing: Label smoothing passed to the cross-entropy.

    Returns:
        Scalar loss.

    Raises:
        ValueError: If ``gamma`` is negative.
    """
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    log_probabilities = F.log_softmax(logits, dim=1)
    log_pt = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1.0 - pt).clamp_min(0.0) ** gamma) * log_pt
    if label_smoothing > 0:
        uniform = -log_probabilities.mean(dim=1)
        loss = (1.0 - label_smoothing) * loss + label_smoothing * uniform
    if weight is not None:
        alpha = weight.to(logits.dtype).to(logits.device)[target]
        loss = alpha * loss
        # Same reduction as ``nn.CrossEntropyLoss(weight=...)``: the weighted
        # sum divided by the sum of the weights, not by the pixel count.  With
        # weights normalised to mean 1 both differ by ``1/mean(alpha)``, which
        # changes the balance against the Dice term.
        return loss.sum() / alpha.sum().clamp_min(1e-6)
    return loss.mean()


class SegLoss(nn.Module):
    """Focal (or plain) cross-entropy plus soft Dice.

    The forward pass returns ``(total, cross_entropy, dice)`` so the training
    log keeps the two terms separate.

    Attributes:
        num_classes: Classes of the problem.
        gamma: Focal parameter; 0 means plain weighted cross-entropy.
        cross_entropy_weight: Weight of the CE/focal term.
        dice_weight: Weight of the Dice term.
    """

    def __init__(
        self,
        num_classes: int,
        weight: torch.Tensor | None = None,
        gamma: float = 0.0,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)
        if weight is None:
            self.register_buffer("weight", torch.ones(num_classes, dtype=torch.float32))
        else:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Compute the three losses.

        Args:
            logits: ``(B, C, H, W)`` model output.
            target: ``(B, H, W)`` int64 classes.

        Returns:
            ``(total, cross_entropy, dice)``.
        """
        weight = self.weight.to(logits.device)
        cross_entropy = focal_cross_entropy(
            logits, target, gamma=self.gamma, weight=weight,
            label_smoothing=self.label_smoothing,
        )
        dice = soft_dice_loss(logits, target, self.num_classes, weight=weight)
        total = self.cross_entropy_weight * cross_entropy + self.dice_weight * dice
        return total, cross_entropy, dice
