"""Cloud / shadow segmentation baseline (G4 / G4b).

Importing this subpackage requires torch (``pip install torch``).  The rest of
:mod:`satharmony` stays NumPy-only.  ``timm`` is only needed when a pretrained
encoder is requested (``ModelConfig(encoder="resnet18", pretrained=True)``).
"""

from .cloud_unet import (
    CHECKPOINT_FORMAT,
    MSS_BAND_NAMES,
    CloudUNet,
    LoadedCheckpoint,
    ModelConfig,
    TimmUNet,
    build_model,
    count_parameters,
    file_sha256,
    load_checkpoint,
    load_model,
    normalize_images,
    predict_masks,
    predict_proba,
    predict_proba_dihedral,
    predict_proba_multiscale,
    save_checkpoint,
)
from .losses import SegLoss, class_weights, focal_cross_entropy, soft_dice_loss
from .metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    apply_class_bias,
    cohen_kappa,
    confusion_matrix,
    evaluate,
    iou_per_class,
    mean_iou,
    overall_accuracy,
    per_class_scores,
    tune_class_bias,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "CLASS_NAMES",
    "MSS_BAND_NAMES",
    "NUM_CLASSES",
    "CloudUNet",
    "LoadedCheckpoint",
    "ModelConfig",
    "SegLoss",
    "TimmUNet",
    "apply_class_bias",
    "build_model",
    "class_weights",
    "cohen_kappa",
    "confusion_matrix",
    "count_parameters",
    "evaluate",
    "file_sha256",
    "focal_cross_entropy",
    "iou_per_class",
    "load_checkpoint",
    "load_model",
    "mean_iou",
    "normalize_images",
    "overall_accuracy",
    "per_class_scores",
    "predict_masks",
    "predict_proba",
    "predict_proba_dihedral",
    "predict_proba_multiscale",
    "save_checkpoint",
    "soft_dice_loss",
    "tune_class_bias",
]
