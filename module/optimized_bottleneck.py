"""
Optimized Sweep Mamba Block and PACM for DermoMamba
These maintain the original architectural properties but fix dimension issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.optimized_mamba import OptimizedSS2D

class OptimizedSweepMamba(nn.Module):
    """
    Optimized Sweep Mamba that adapts to any spatial dimensions
    Maintains the same multi-directional processing but with robust dimensions
    """
    def __init__(self, dim, ratio=8):
        super().__init__()
        self.dim = dim
        self.reduced_dim = max(dim // ratio, 16)  # Ensure minimum dimension
        
        # Normalization and projections
        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, self.reduced_dim, bias=False)
        
        # Multiple Mamba blocks for different sweep directions
        self.mamba_horizontal = OptimizedSS2D(
            d_model=self.reduced_dim, 
            dropout=0, 
            d_state=16
        )
        self.mamba_vertical = OptimizedSS2D(
            d_model=self.reduced_dim, 
            dropout=0, 
            d_state=16
        )
        self.mamba_diagonal = OptimizedSS2D(
            d_model=self.reduced_dim, 
            dropout=0, 
            d_state=16
        )
        
        # Activation and output projection
        self.act = nn.SiLU()
        self.relu = nn.ReLU(inplace=True)
        self.proj_out = nn.Linear(self.reduced_dim, dim, bias=False)
        
        # Learnable scaling and normalization
        self.scale = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(dim)
        
        # Mixing weights for different directions
        self.mixing_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to (B, H, W, C) format
        x = x.permute(0, 2, 3, 1)
        skip = x
        
        # Project to reduced dimension
        x = self.proj_in(self.ln(x))  # (B, H, W, reduced_dim)
        
        # Process in different sweep directions
        
        # 1. Horizontal sweep (left-to-right)
        x_flat = x.view(B, H * W, self.reduced_dim)
        x1 = self.mamba_horizontal(x_flat)
        x1 = x1.view(B, H, W, self.reduced_dim)
        
        # 2. Vertical sweep (top-to-bottom) 
        x_transposed = x.permute(0, 2, 1, 3)  # (B, W, H, reduced_dim)
        x_flat = x_transposed.contiguous().view(B, W * H, self.reduced_dim)
        x2 = self.mamba_vertical(x_flat)
        x2 = x2.view(B, W, H, self.reduced_dim).permute(0, 2, 1, 3)  # Back to (B, H, W, reduced_dim)
        
        # 3. Diagonal sweep (approximate with reshaped processing)
        # For robustness, we'll use a different reshaping strategy
        if H * W >= self.reduced_dim:
            x_reshaped = x.view(B, -1, self.reduced_dim)
            x3 = self.mamba_diagonal(x_reshaped)
            x3 = x3.view(B, H, W, self.reduced_dim)
        else:
            # Fallback for very small spatial dimensions
            x3 = x
        
        # Combine different sweep directions with learnable weights
        w = torch.softmax(self.mixing_weights, dim=0)
        combined = w[0] * x1 + w[1] * x2 + w[2] * x3
        
        # Apply activation and gating
        gate = self.act(combined)
        out = gate * combined
        
        # Project back to original dimension
        out = self.proj_out(out)
        
        # Residual connection with learnable scaling
        out = out + skip * self.scale
        
        # Convert back to (B, C, H, W) format
        out = out.permute(0, 3, 1, 2)
        
        # Apply batch norm and activation
        out = self.relu(self.bn(out))
        
        return out

class OptimizedPCAM(nn.Module):
    """
    Optimized Pyramid Channel Attention Module
    Maintains the original multi-scale channel attention properties
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-scale channel attention with different kernel sizes
        reduced_dim = max(dim // reduction, 8)
        
        # Channel attention branches
        self.fc1 = nn.Sequential(
            nn.Linear(dim, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim, bias=False)
        )
        
        # Spatial attention with different scales
        self.spatial_att1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.spatial_att2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.spatial_att3 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        
        # Combine spatial attentions
        self.spatial_combine = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable combination weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel attention
        y = self.global_pool(x).view(B, C)  # Global average pooling
        y = self.fc1(y).view(B, C, 1, 1)    # Channel attention weights
        channel_att = self.sigmoid(y)
        
        # Spatial attention
        spatial_input = torch.mean(x, dim=1, keepdim=True)  # Average across channels
        
        # Multi-scale spatial attention
        s1 = self.spatial_att1(spatial_input)
        s2 = self.spatial_att2(spatial_input)
        s3 = self.spatial_att3(spatial_input)
        
        # Combine spatial attentions
        spatial_combined = torch.cat([s1, s2, s3], dim=1)
        spatial_att = self.sigmoid(self.spatial_combine(spatial_combined))
        
        # Apply attentions
        out = x * channel_att * self.alpha + x * spatial_att * self.beta
        
        return out

# Alias for compatibility
OptimizedPCA = OptimizedPCAM
