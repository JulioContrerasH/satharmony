"""Compact U-Net baseline for 4-band MSS cloud / shadow segmentation.

This is the G4 baseline of the ``cloud-mss-esa`` plan: a small U-Net trained on
the pseudo-MSS fixture built from CloudSEN12 (G3) and evaluated zero-shot on
the five real PICS.  It predicts the four CloudSEN12 classes
(``0 clear``, ``1 thick cloud``, ``2 thin cloud``, ``3 cloud shadow``) from the
four MSS bands (green, red, NIR1, NIR2) in top-of-atmosphere reflectance.

Torch is not a core dependency of the package: install it separately
(``pip install torch``).  Only this submodule imports it, so ``import
satharmony`` keeps working in a numpy-only environment.

The checkpoints written here are plain dictionaries with ``weights_only=True``
compatible content (no pickled NumPy objects), plus a SHA-256 sidecar so the
exact file used by a report can be verified later.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

#: Version tag stored in every checkpoint.
CHECKPOINT_FORMAT = "satharmony-cloud-unet/1"

#: MSS band order of the model input.
MSS_BAND_NAMES: tuple[str, ...] = ("green", "red", "nir1", "nir2")


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in blocks.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the baseline.

    Attributes:
        in_channels: Input bands (4 for Landsat MSS).
        num_classes: Output classes (4, CloudSEN12 encoding).
        base_channels: Width of the first encoder level; doubles per level.
        depth: Downsampling steps. The input must be a multiple of
            ``2 ** depth``.
        norm_groups: Groups of the GroupNorm layers; must divide every
            channel count.
        dropout: Spatial dropout applied to the bottleneck, ``0`` disables it.
    """

    in_channels: int = 4
    num_classes: int = 4
    base_channels: int = 32
    depth: int = 4
    norm_groups: int = 8
    dropout: float = 0.0
    encoder: str | None = None
    pretrained: bool = False

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.num_classes < 2:
            raise ValueError("in_channels must be >= 1 and num_classes >= 2")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.base_channels < 1 or self.base_channels % self.norm_groups:
            raise ValueError(
                f"base_channels={self.base_channels} must be a multiple of "
                f"norm_groups={self.norm_groups}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def size_multiple(self) -> int:
        """Input size must be a multiple of this."""
        return 2**self.depth if self.encoder is None else 32


class DoubleConv(nn.Module):
    """``Conv3x3 -> GroupNorm -> SiLU`` twice."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool by 2 followed by a :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate the skip and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm_groups: int = 8) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = DoubleConv(out_channels + skip_channels, out_channels, norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.block(torch.cat([x, skip], dim=1))


class CloudUNet(nn.Module):
    """Small U-Net: 4 MSS bands in, 4 cloud classes out (logits)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        groups = cfg.norm_groups

        channels = [cfg.base_channels * 2**level for level in range(cfg.depth)]
        self.stem = DoubleConv(cfg.in_channels, channels[0], groups)
        self.downs = nn.ModuleList(
            Down(channels[level - 1], channels[level], groups)
            for level in range(1, cfg.depth)
        )
        self.bottleneck = Down(channels[-1], channels[-1] * 2, groups)
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.ups = nn.ModuleList()
        for level in reversed(range(cfg.depth)):
            # The decoder input at every level is twice the level width: the
            # bottleneck (2 * channels[-1]) or the previous decoder output.
            self.ups.append(Up(channels[level] * 2, channels[level], channels[level], groups))
        self.head = nn.Conv2d(channels[0], cfg.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(N, in_channels, H, W)`` tensor; ``H`` and ``W`` multiples of
                ``2 ** depth``.

        Returns:
            ``(N, num_classes, H, W)`` logits.

        Raises:
            ValueError: If the spatial size is not a multiple of
                ``2 ** depth``.
        """
        multiple = self.config.size_multiple
        if x.shape[-1] % multiple or x.shape[-2] % multiple:
            raise ValueError(f"input size {tuple(x.shape[-2:])} is not a multiple of {multiple}")

        skips = []
        features = self.stem(x)
        skips.append(features)
        for down in self.downs:
            features = down(features)
            skips.append(features)
        features = self.dropout(self.bottleneck(features))

        for level, up in enumerate(self.ups):
            features = up(features, skips[-1 - level])
        return self.head(features)


class TimmUNet(nn.Module):
    """U-Net around a ``timm`` encoder, optionally with ImageNet weights.

    The four finest encoder levels (strides 4, 8, 16 and 32) feed the decoder;
    the full-resolution stem skip comes from a small convolution on the input
    bands, so the output keeps the input resolution.  ``timm`` is imported
    lazily: the package still works without it unless an ``encoder`` is
    requested.

    Attributes:
        config: Architecture used to build the model.
        encoder: The ``timm`` feature extractor.
        stem: Full-resolution skip.
        ups: Decoder blocks, coarse to fine.
        head: ``1x1`` convolution to the class logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.encoder:
            raise ValueError("TimmUNet needs config.encoder")
        try:
            import timm
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("pip install timm to use a pretrained encoder") from error

        self.config = config
        self.encoder = timm.create_model(
            config.encoder,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            features_only=True,
        )
        channels = list(self.encoder.feature_info.channels())
        reductions = list(self.encoder.feature_info.reduction())
        if len(channels) < 5 or reductions[-5:] != [2, 4, 8, 16, 32]:
            raise ValueError(
                f"encoder {config.encoder} must provide strides 2, 4, 8, 16 and 32; "
                f"got {reductions}"
            )
        fine_channels = channels[-5:]

        groups = config.norm_groups
        stem_channels = config.base_channels
        self.stem = DoubleConv(config.in_channels, stem_channels, groups)
        self.reduce = nn.ModuleList(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False) for channel in fine_channels
        )
        self.ups = nn.ModuleList()
        decoder_in = fine_channels[-1]
        for index in range(4, -1, -1):
            skip_channels = fine_channels[index - 1] if index > 0 else stem_channels
            self.ups.append(Up(decoder_in, skip_channels, skip_channels, groups))
            decoder_in = skip_channels
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(decoder_in, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(N, in_channels, H, W)`` tensor; ``H`` and ``W`` multiples of 32.

        Returns:
            ``(N, num_classes, H, W)`` logits.

        Raises:
            ValueError: If the spatial size is not a multiple of 32.
        """
        multiple = self.config.size_multiple
        if x.shape[-1] % multiple or x.shape[-2] % multiple:
            raise ValueError(f"input size {tuple(x.shape[-2:])} is not a multiple of {multiple}")

        skips = [
            self.reduce[index](feature)
            for index, feature in enumerate(self.encoder(x)[-5:])
        ]
        features = self.dropout(skips[-1])
        stem = self.stem(x)
        for index, up in enumerate(self.ups):
            skip = stem if index == len(self.ups) - 1 else skips[3 - index]
            features = up(features, skip)
        return self.head(features)


def build_model(config: ModelConfig | None = None) -> nn.Module:
    """Build the segmentation model with fresh weights.

    Args:
        config: Architecture; defaults to :class:`ModelConfig`.

    Returns:
        :class:`CloudUNet` for ``encoder=None``, :class:`TimmUNet` otherwise,
        in training mode.
    """
    config = config or ModelConfig()
    if config.encoder is None:
        return CloudUNet(config)
    return TimmUNet(config)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_images(
    images: NDArray[np.floating],
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Standardise reflectance with the training statistics.

    Args:
        images: ``(..., 4, H, W)`` reflectance, 0-1.
        band_mean: Per-band mean, shape ``(4,)``.
        band_std: Per-band standard deviation, shape ``(4,)``.

    Returns:
        float32 array of the same shape.
    """
    mean = np.asarray(band_mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(band_std, dtype=np.float32).reshape(-1, 1, 1)
    if std.size != images.shape[-3]:
        raise ValueError(f"statistics have {std.size} bands, image has {images.shape[-3]}")
    return ((np.asarray(images, dtype=np.float32) - mean) / std).astype(np.float32)


def _pad_to_multiple(images: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad ``(N, C, H, W)`` to a multiple of ``multiple``."""
    height, width = images.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        images = np.pad(images, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return images, (pad_h, pad_w)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    images: NDArray[np.floating],
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> NDArray[np.float32]:
    """Class probabilities for a stack of images.

    Args:
        model: A :class:`CloudUNet` (or compatible) already built.
        images: ``(N, C, H, W)`` or ``(C, H, W)`` reflectance, 0-1.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        batch_size: Images per forward pass.
        device: Torch device.

    Returns:
        ``(N, num_classes, H, W)`` probabilities.
    """
    single = images.ndim == 3
    batch = images[None] if single else images
    batch = normalize_images(batch, band_mean, band_std)
    batch, _ = _pad_to_multiple(batch, model.config.size_multiple)  # type: ignore[attr-defined]

    was_training = model.training
    model.eval()
    outputs = []
    for start in range(0, batch.shape[0], batch_size):
        chunk = torch.from_numpy(batch[start : start + batch_size]).to(device)
        logits = model(chunk)
        outputs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    if was_training:
        model.train()

    probabilities = np.concatenate(outputs, axis=0)
    height, width = images.shape[-2:]
    probabilities = probabilities[:, :, :height, :width]
    return probabilities[0] if single else probabilities


@torch.no_grad()
def predict_masks(
    model: nn.Module,
    images: NDArray[np.floating],
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> NDArray[np.uint8]:
    """Argmax class per pixel.

    Args:
        model: Model.
        images: ``(N, C, H, W)`` or ``(C, H, W)`` reflectance, 0-1.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        batch_size: Images per forward pass.
        device: Torch device.

    Returns:
        ``(N, H, W)`` or ``(H, W)`` uint8 classes.
    """
    probabilities = predict_proba(
        model, images, band_mean, band_std, batch_size=batch_size, device=device
    )
    return probabilities.argmax(axis=-3).astype(np.uint8)


def _resize_batch(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a ``(N, C, H, W)`` float array."""
    tensor = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.numpy()


def predict_proba_dihedral(
    model: nn.Module,
    images: NDArray[np.floating],
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> NDArray[np.float32]:
    """Average the softmax over the eight dihedral transforms.

    Flips and 90-degree rotations are exact symmetries of a top-of-atmosphere
    image, so the average reduces the prediction variance without any training
    cost.  Inference is eight times slower than a single pass.

    Args:
        model: Trained model.
        images: ``(N, C, H, W)`` or ``(C, H, W)`` reflectance, 0-1.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        batch_size: Images per forward pass.
        device: Torch device.

    Returns:
        Probabilities with the input shape (``(N, num_classes, H, W)`` or
        ``(num_classes, H, W)``).
    """
    single = images.ndim == 3
    batch = np.ascontiguousarray(images[None] if single else images, dtype=np.float32)
    accumulated = None
    for rotations in range(4):
        for flip in (False, True):
            transformed = np.rot90(batch, rotations, axes=(2, 3))
            if flip:
                transformed = transformed[:, :, :, ::-1]
            probabilities = predict_proba(
                model,
                np.ascontiguousarray(transformed),
                band_mean,
                band_std,
                batch_size=batch_size,
                device=device,
            )
            if flip:
                probabilities = probabilities[:, :, :, ::-1]
            probabilities = np.rot90(probabilities, -rotations, axes=(2, 3))
            accumulated = probabilities if accumulated is None else accumulated + probabilities
    result = accumulated / 8.0
    return result[0] if single else result


def predict_proba_multiscale(
    model: nn.Module,
    images: NDArray[np.floating],
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
    scales: tuple[float, ...] = (1.0,),
    dihedral: bool = True,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> NDArray[np.float32]:
    """Average the probabilities over several input scales.

    The network is fully convolutional, so the chip can be resized before the
    forward pass and the probabilities resized back.  Rescaling changes the
    effective receptive field relative to the objects, which helps the thin
    cloud pixels that are barely resolved at 60 m.

    Args:
        model: Trained model.
        images: ``(N, C, H, W)`` or ``(C, H, W)`` reflectance, 0-1.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        scales: Scale factors; ``1.0`` is the training resolution.
        dihedral: Average the eight dihedral transforms inside every scale.
        batch_size: Images per forward pass.
        device: Torch device.

    Returns:
        Probabilities with the input shape, renormalised to sum 1 per pixel.

    Raises:
        ValueError: If ``scales`` is empty or contains a non-positive value.
    """
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must be a non-empty tuple of positive values")
    single = images.ndim == 3
    batch = np.ascontiguousarray(images[None] if single else images, dtype=np.float32)
    height, width = batch.shape[-2:]
    accumulated = None
    for scale in scales:
        if scale == 1.0:
            scaled = batch
        else:
            target_h = max(model.config.size_multiple, round(height * scale))  # type: ignore[attr-defined]
            target_w = max(model.config.size_multiple, round(width * scale))  # type: ignore[attr-defined]
            scaled = _resize_batch(batch, target_h, target_w)
        if dihedral:
            probabilities = predict_proba_dihedral(
                model, scaled, band_mean, band_std, batch_size=batch_size, device=device
            )
        else:
            probabilities = predict_proba(
                model, scaled, band_mean, band_std, batch_size=batch_size, device=device
            )
        if scale != 1.0:
            probabilities = _resize_batch(probabilities, height, width)
        accumulated = probabilities if accumulated is None else accumulated + probabilities
    result = accumulated / len(scales)
    result = result / np.maximum(result.sum(axis=-3, keepdims=True), 1e-8)
    return result[0] if single else result


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ModelConfig,
    band_mean: NDArray[np.floating],
    band_std: NDArray[np.floating],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a checkpoint plus its SHA-256 sidecar.

    The payload holds only JSON-friendly values and tensors, so it can be read
    back with ``torch.load(..., weights_only=True)``.

    Args:
        path: Destination ``.pt`` file.
        model: Trained model.
        config: Architecture used to build it.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        metadata: Extra JSON-friendly information (epoch, metrics, data hash).

    Returns:
        Dict with ``path``, ``sha256`` and ``bytes``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": asdict(config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "band_mean": np.asarray(band_mean, dtype=np.float32).tolist(),
        "band_std": np.asarray(band_std, dtype=np.float32).tolist(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    digest = file_sha256(path)
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Result of :func:`load_checkpoint`."""

    config: ModelConfig
    state_dict: dict[str, torch.Tensor]
    band_mean: NDArray[np.float32]
    band_std: NDArray[np.float32]
    metadata: dict[str, Any]
    sha256: str


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: Torch map location.

    Returns:
        The architecture, weights, normalisation statistics, metadata and the
        SHA-256 of the file on disk.

    Raises:
        ValueError: If the payload is not a recognised checkpoint.
        KeyError: If a required field is missing.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    config = ModelConfig(**payload["config"])
    return LoadedCheckpoint(
        config=config,
        state_dict=payload["state_dict"],
        band_mean=np.asarray(payload["band_mean"], dtype=np.float32),
        band_std=np.asarray(payload["band_std"], dtype=np.float32),
        metadata=dict(payload.get("metadata", {})),
        sha256=file_sha256(path),
    )


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[CloudUNet, LoadedCheckpoint]:
    """Build the model and load the weights of a checkpoint.

    Args:
        path: Checkpoint file.
        device: Device the model is moved to.

    Returns:
        ``(model, checkpoint)``; the model is in eval mode.
    """
    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint.config)
    model.load_state_dict(checkpoint.state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint
