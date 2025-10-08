"""KAN-based coupling subnetwork implementations."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch import nn

from kan import KAN


class KANCouplingNet(nn.Module):
    """Per-pixel Kolmogorov--Arnold Network used in affine coupling blocks.

    The module flattens the spatial dimensions so that each pixel's feature
    vector is processed independently by a fully-connected KAN model and then
    restores the original tensor layout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: Iterable[int] | None = None,
        *,
        grid: int = 5,
        k: int = 3,
        symbolic_enabled: bool = False,
        enable_speed: bool = True,
        identity_init: bool = False,
        identity_jitter: float = 1e-3,
        auto_save: bool = False,
        ckpt_path: Optional[str] = None,
        verbose: bool = False,
        seed: Optional[int] = 42,
        chunk_size: Optional[int] = None,
        normalize_input: bool = False,
        normalization_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        hidden_list: Sequence[int]
        if hidden_dims is None:
            hidden_list = (64, 64)
        else:
            hidden_list = tuple(hidden_dims)

        width = [in_channels, *hidden_list, out_channels]
        kan_kwargs = {
            "width": width,
            "grid": grid,
            "k": k,
            "symbolic_enabled": symbolic_enabled,
            "auto_save": auto_save,
        }
        if ckpt_path is not None:
            kan_kwargs["ckpt_path"] = ckpt_path

        self.kan = KAN(**kan_kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.normalize_input = bool(normalize_input)
        self.normalization_eps = float(normalization_eps)
        if enable_speed:
            if hasattr(self.kan, "speed"):
                # ``speed`` disables the symbolic branch for improved performance.
                self.kan.speed()
            elif hasattr(self.kan, "enable_speed"):
                # Backwards compatibility with older pykan versions.
                self.kan.enable_speed()

        if identity_init:
            jitter = float(identity_jitter)
            with torch.no_grad():
                for parameter in self.kan.parameters():
                    parameter.zero_()
                    if parameter.dim() > 1 and jitter > 0.0:
                        parameter.add_(jitter * torch.randn_like(parameter))

        if verbose:
            print(
                "[KANCouplingNet] Initialized",
                {
                    "in": in_channels,
                    "out": out_channels,
                    "hidden": hidden_list,
                    "grid": grid,
                    "k": k,
                    "seed": seed,
                },
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the KAN to every spatial location independently."""
        if x.dim() != 4:
            raise ValueError(
                "KANCouplingNet expects a 4D tensor of shape (N, C, H, W); "
                f"got {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                "Channel mismatch: expected input with "
                f"{self.in_channels} channels, got {channels}"
            )

        flattened = x.movedim(1, -1).reshape(-1, channels).contiguous()
        if self.normalize_input:
            mean = flattened.mean(dim=-1, keepdim=True)
            std = flattened.std(dim=-1, keepdim=True)
            std = std.clamp_min(self.normalization_eps)
            flattened = (flattened - mean) / std
        if self.chunk_size is None or flattened.size(0) <= self.chunk_size:
            transformed = self.kan(flattened)
        else:
            chunks = []
            for chunk in flattened.split(self.chunk_size, dim=0):
                chunks.append(self.kan(chunk))
            transformed = torch.cat(chunks, dim=0)
        if transformed.shape[-1] != self.out_channels:
            raise RuntimeError(
                "KAN output shape mismatch: expected last dim "
                f"{self.out_channels}, got {transformed.shape[-1]}"
            )
        restored = transformed.view(batch_size, height, width, self.out_channels).movedim(-1, 1)
        return restored