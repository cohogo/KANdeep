"""Perceptual loss helpers built on a frozen VGG-19 feature extractor.

This module exposes :class:`VGGLoss`, which matches the interface used by the
original DeepMIH training scripts.  The implementation wraps the VGG-19 network
from :mod:`torchvision` and returns intermediate features so that the calling
code can measure distances between perceptual embeddings.  The helper keeps the
feature extractor frozen and normalises the input tensors with the ImageNet
statistics expected by VGG models.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import nn
from torchvision import models

__all__ = ["VGGLoss"]


def _resolve_vgg_weights() -> Optional[models.VGG19_Weights]:
    """Return the default VGG-19 weights if available.

    Older versions of :mod:`torchvision` exposed ``pretrained=True`` instead of
    the weights enum.  The helper returns ``None`` when the enum is missing so
    we can gracefully fall back to the legacy API.
    """

    weights_enum = getattr(models, "VGG19_Weights", None)
    if weights_enum is None:
        return None
    return getattr(weights_enum, "IMAGENET1K_V1", None)


class VGGLoss(nn.Module):
    """Frozen VGG-19 feature extractor for perceptual losses.

    Parameters
    ----------
    input_nc:
        Unused placeholder kept for compatibility with the legacy training
        scripts.  The DeepMIH code always instantiates the loss with ``3``.
    output_nc:
        Unused compatibility placeholder.  Retained to mirror the signature of
        the historical implementation.
    requires_grad:
        When ``True`` the VGG parameters remain trainable, otherwise they are
        frozen.  The DeepMIH scripts pass ``False`` so the extractor acts as a
        fixed feature network.
    layer:
        Name of the VGG activation to return.  The default ``"relu3_3"``
        matches the behaviour of many perceptual-loss implementations and keeps
        the tensor size manageable.
    """

    _LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,
        "relu3_4": 18,
        "relu4_3": 25,
        "relu4_4": 27,
        "relu5_4": 36,
    }

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        requires_grad: bool = False,
        layer: str = "relu3_3",
    ) -> None:
        super().__init__()

        if layer not in self._LAYER_MAP:
            raise ValueError(f"Unsupported VGG layer '{layer}'. Available keys: {sorted(self._LAYER_MAP)}")

        layer_idx = self._LAYER_MAP[layer]
        weights = _resolve_vgg_weights()

        try:
            if weights is not None:
                vgg = models.vgg19(weights=weights).features[: layer_idx + 1]
            else:
                vgg = models.vgg19(pretrained=True).features[: layer_idx + 1]
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.warn(f"Falling back to randomly initialised VGG-19 weights: {exc}")
            vgg = models.vgg19(weights=None).features[: layer_idx + 1]

        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad_(False)

        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return VGG features for ``x``.

        The tensor is normalised with ImageNet statistics.  Inputs with a single
        channel are broadcast to three channels so grayscale tensors can reuse
        the same extractor.
        """

        if x.dim() != 4:
            raise ValueError(f"Expected a 4D tensor (NCHW) but received shape {tuple(x.shape)}")

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
            raise ValueError("VGGLoss expects inputs with 1 or 3 channels")

        x = (x - self.mean) / self.std
        return self.vgg(x)