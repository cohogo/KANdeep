import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from typing import Optional

try:
    from timm.models.vision_transformer import VisionTransformer, _cfg, Block, Attention, Mlp
    from timm.models.layers import DropPath
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm is not available. VisionKAN functionality will be limited.")

from fasterkan import FasterKAN


class MoE_KAN_MLP(nn.Module):
    """
    Mixture of Experts KAN MLP implementation.
    
    This implementation is strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, ffn_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts  # default = 8
        self.top_k = top_k  # default = 2

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Half traditional MLP experts, half KAN experts
        if TIMM_AVAILABLE:
            self.experts = nn.ModuleList([
                Mlp(in_features=self.hidden_dim, hidden_features=self.ffn_dim) 
                for _ in range(self.num_experts//2)
            ])
        else:
            # Fallback to simple linear layers if timm is not available
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.ffn_dim),
                    nn.GELU(),
                    nn.Linear(self.ffn_dim, self.hidden_dim)
                ) for _ in range(self.num_experts//2)
            ])

        # Add KAN experts
        for _ in range(self.num_experts//2):
            kan = FasterKAN([self.hidden_dim, self.ffn_dim//2, self.hidden_dim])
            self.experts.append(kan)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE KAN MLP."""
        # Add batch_size dimension if needed
        original_shape = hidden_states.shape
        if len(original_shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = routing_weights[top_x_list, idx_list, None] * current_hidden_states
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # Remove batch_size dimension if it was added
        if len(original_shape) == 2:
            final_hidden_states = final_hidden_states.squeeze(0)

        return final_hidden_states


if TIMM_AVAILABLE:
    class kanBlock(Block):
        """KAN Block that replaces the MLP in Vision Transformer with MoE KAN MLP."""

        def __init__(self, dim, num_heads=8, hdim_kan=192, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                     drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__(dim, num_heads)
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            # Replace standard MLP with MoE KAN MLP
            self.kan = MoE_KAN_MLP(hidden_dim=dim, ffn_dim=hdim_kan, num_experts=8, top_k=2)

        def forward(self, x):
            b, t, d = x.shape
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))
            return x


    class VisionKAN(VisionTransformer):
        """Vision Transformer with KAN layers."""
        
        def __init__(self, *args, num_heads=8, batch_size=16, **kwargs):
            if 'hdim_kan' in kwargs:
                self.hdim_kan = kwargs['hdim_kan']
                del kwargs['hdim_kan']
            else:
                self.hdim_kan = 192
            
            super().__init__(*args, **kwargs)
            self.num_heads = num_heads
            
            # For newer version timm they don't save the depth to self.depth, so we need to check it
            try:
                self.depth
            except AttributeError:
                if 'depth' in kwargs:
                    self.depth = kwargs['depth']
                else:
                    self.depth = 12

            block_list = [
                kanBlock(dim=self.embed_dim, num_heads=self.num_heads, hdim_kan=self.hdim_kan)
                for i in range(self.depth)
            ]
            
            # Check the origin type of the block is torch.nn.modules.container.Sequential
            # if the origin type is torch.nn.modules.container.Sequential, then we need to convert it to a list
            if type(self.blocks) == nn.Sequential:
                self.blocks = nn.Sequential(*block_list)
            elif type(self.blocks) == nn.ModuleList:
                self.blocks = nn.ModuleList(block_list)
else:
    # Dummy classes when timm is not available
    class kanBlock(nn.Module):
        def __init__(self, dim, num_heads=8, hdim_kan=192, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                     drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            print("Warning: kanBlock requires timm. This is a dummy implementation.")
        
        def forward(self, x):
            return x
    
    class VisionKAN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Warning: VisionKAN requires timm. This is a dummy implementation.")
        
        def forward(self, x):
            return x


def create_kan(model_name: str, pretrained: bool = False, **kwargs):
    """
    Create a KAN-based Vision Transformer model.
    
    Args:
        model_name: Name of the model to create
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments for model creation
        
    Returns:
        VisionKAN model instance
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for VisionKAN functionality. Please install it with: pip install timm")
    
    if model_name == 'deit_tiny_patch16_224_KAN':
        model = VisionKAN(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        return model
    
    elif model_name == 'deit_small_patch16_224_KAN':
        model = VisionKAN(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_224_KAN':
        model = VisionKAN(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_384_KAN':
        model = VisionKAN(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")