"""
Simplified VSSBlock implementation for DermoMamba
This replaces the original VSSBlock.py with a version that doesn't require mamba-ssm
"""
import torch
import torch.nn as nn
import math
from einops import rearrange
from module.simplified_mamba import SimplifiedSS2D

class SS2D(SimplifiedSS2D):
    """Alias for SimplifiedSS2D to maintain compatibility"""
    pass

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # input: (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = input.shape
        input_flat = input.view(B, H * W, C)
        
        # Apply layer norm and self attention
        x = input_flat + self.drop_path(self.self_attention(self.ln_1(input_flat)))
        
        # Reshape back to (B, H, W, C)
        return x.view(B, H, W, C)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
