"""LoKi-style KAN adapter blocks for DeepMIH coupling sub-networks.

This module introduces two building blocks:

* :class:`KANLayer` implements a lightweight Kolmogorov--Arnold layer where every edge
  is parameterised by a learnable residual spline activation.  The formulation mirrors
  the definition from the KAN paper but keeps the grid compact (small number of basis
  functions) so that it can act as a drop-in replacement for the tiny MLPs used in the
  coupling blocks.
* :class:`LoKiKANAdapter` wraps two KAN layers between a linear bottleneck/expansion pair,
  matching the LoKi adapter design that serves as a learnable activation while
  controlling memory usage.

The classes are self-contained and can be imported without altering existing training
scripts.  A later step will swap the stage-2 scale subnetworks to use the adapter.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KANLayer", "LoKiKANAdapter"]


class KANLayer(nn.Module):
    """Minimal Kolmogorov--Arnold layer with residual spline activations.

    The layer preserves the fully-connected topology of a dense layer but replaces each
    edge weight with a learnable 1D spline that modulates a SiLU residual branch.  It is
    intentionally compact (small grid size, low order) to keep GPU memory usage low.

    Args:
        in_features:  Number of input channels.
        out_features: Number of output channels.
        grid_size:    Number of interior grid cells for the spline basis.
        order:        Spline order (``order=3`` corresponds to cubic B-splines).
        normalize_input: Whether to keep running min/max statistics so that inputs stay
            inside the grid support.  This mimics the stabilisation trick described in
            KAN literature.
        momentum:     Update factor for the running statistics when normalisation is on.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 4,
        order: int = 3,
        normalize_input: bool = True,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("KANLayer requires positive feature dimensions.")
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1")
        if order < 0:
            raise ValueError("order must be >= 0")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.order = order
        self.normalize_input = normalize_input
        self.momentum = momentum
        # Number of basis functions for the spline component.
        self.num_basis = grid_size + order

        # Edge parameters.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_coeff = nn.Parameter(torch.empty(out_features, in_features, self.num_basis))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Buffers for input range tracking.
        if normalize_input:
            self.register_buffer("running_min", torch.zeros(in_features))
            self.register_buffer("running_max", torch.ones(in_features))
            self.register_buffer("_stats_initialized", torch.tensor(False))
        else:
            self.register_buffer("running_min", None)
            self.register_buffer("running_max", None)
            self.register_buffer("_stats_initialized", torch.tensor(True))

        # Pre-computed knot vector for a uniform spline grid over [-1, 1].
        knot_count = self.num_basis + self.order + 1
        knots = torch.linspace(-1.0, 1.0, knot_count)
        self.register_buffer("base_knots", knots)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.constant_(self.spline_weight, 1.0)
        nn.init.normal_(self.spline_coeff, mean=0.0, std=1e-3)
        nn.init.zeros_(self.bias)
        if self.normalize_input:
            self.running_min.zero_()
            self.running_max.fill_(1.0)
            self._stats_initialized.fill_(False)

    def _normalise_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return x
        if self.training:
            with torch.no_grad():
                current_min = x.amin(dim=0)
                current_max = x.amax(dim=0)
                if not bool(self._stats_initialized):
                    self.running_min.copy_(current_min)
                    self.running_max.copy_(current_max)
                    self._stats_initialized.fill_(True)
                else:
                    self.running_min.lerp_(current_min, self.momentum)
                    self.running_max.lerp_(current_max, self.momentum)
        if bool(self._stats_initialized):
            min_val = self.running_min
            max_val = self.running_max
        else:
            min_val = x.amin(dim=0)
            max_val = x.amax(dim=0)

        width = (max_val - min_val).clamp_min(1e-2)
        centre = 0.5 * (max_val + min_val)
        scaled = (x - centre) / (0.5 * width)
        return scaled.clamp_(-2.0, 2.0)

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions for the given inputs.

        Args:
            x: Normalised inputs with shape ``(batch, in_features)``.

        Returns:
            Tensor of shape ``(batch, in_features, num_basis)`` containing the basis
            activations for every input channel.
        """

        device = x.device
        knots = self.base_knots.to(device)
        num_basis = self.num_basis
        order = self.order

        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)

        # Order-0 basis functions (piecewise constants on knot intervals).
        basis = []
        for i in range(num_basis):
            left = knots[i]
            right = knots[i + 1]
            support = ((x_expanded >= left) & (x_expanded < right)).to(x.dtype)
            if i == num_basis - 1:
                support = torch.where(x_expanded == right, torch.ones_like(support), support)
            basis.append(support)
        basis_tensor = torch.stack(basis, dim=-1)

        # Recursive Cox–de Boor formula to reach the requested order.
        for k in range(1, order + 1):
            updated = []
            for i in range(num_basis):
                denom1 = knots[i + k] - knots[i]
                denom2 = knots[i + k + 1] - knots[i + 1]

                if denom1 > 0:
                    term1 = (x_expanded - knots[i]) / denom1 * basis_tensor[..., i]
                else:
                    term1 = torch.zeros_like(x_expanded)

                if i + 1 < num_basis and denom2 > 0:
                    next_basis = basis_tensor[..., i + 1]
                    term2 = (knots[i + k + 1] - x_expanded) / denom2 * next_basis
                else:
                    term2 = torch.zeros_like(x_expanded)

                updated.append(term1 + term2)
            basis_tensor = torch.stack(updated, dim=-1)
        return basis_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if original_shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {original_shape[-1]}"
            )
        x2d = x.reshape(-1, self.in_features)
        x_norm = self._normalise_input(x2d)
        basis = self._bspline_basis(x_norm)
        silu_x = F.silu(x_norm)

        base_vals = silu_x.unsqueeze(1) * self.base_weight.unsqueeze(0)
        spline_vals = torch.einsum("bim,oim->boi", basis, self.spline_coeff)
        spline_vals = spline_vals * self.spline_weight.unsqueeze(0)

        edge_outputs = base_vals + spline_vals
        out = edge_outputs.sum(dim=2) + self.bias.unsqueeze(0)
        out = out.view(*original_shape[:-1], self.out_features)
        return out


class LoKiKANAdapter(nn.Module):
    """Two-layer KAN adapter with linear bottleneck/expansion.

    The adapter mirrors the LoKi design: a linear down-projection, two KAN layers acting
    as a learnable activation in the bottleneck space, and a linear up-projection back to
    the original feature dimension.  Optional layer normalisation and dropout can be
    enabled to match surrounding Transformer-style blocks if necessary.
    """

    def __init__(
        self,
        d_in: int,
        *,
        bottleneck_ratio: float = 1.0 / 16.0,
        hidden_features: Optional[int] = 32,
        grid_size: int = 4,
        order: int = 3,
        normalize_input: bool = True,
        activation_dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_in <= 0:
            raise ValueError("d_in must be positive")
        if not 0 < bottleneck_ratio <= 1:
            raise ValueError("bottleneck_ratio must be in (0, 1]")
        bottleneck_dim = max(1, int(round(d_in * bottleneck_ratio)))
        if hidden_features is not None and hidden_features <= 0:
            raise ValueError("hidden_features must be positive when provided")

        self.down = nn.Linear(d_in, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim) if use_layer_norm else None
        kan_hidden = hidden_features or bottleneck_dim
        self.kan1 = KANLayer(
            bottleneck_dim,
            kan_hidden,
            grid_size=grid_size,
            order=order,
            normalize_input=normalize_input,
        )
        self.kan2 = KANLayer(
            kan_hidden,
            bottleneck_dim,
            grid_size=grid_size,
            order=order,
            normalize_input=normalize_input,
        )
        self.up = nn.Linear(bottleneck_dim, d_in)
        self.dropout = nn.Dropout(activation_dropout) if activation_dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if original_shape[-1] != self.down.in_features:
            raise ValueError(
                f"Expected last dimension {self.down.in_features}, got {original_shape[-1]}"
            )
        x2d = x.reshape(-1, original_shape[-1])
        z = self.down(x2d)
        if self.norm is not None:
            z = self.norm(z)
        z = self.kan1(z)
        z = self.dropout(F.silu(z))
        z = self.kan2(z)
        z = self.dropout(F.silu(z))
        out = self.up(z)
        out = out.view(*original_shape[:-1], self.down.in_features)
        return out