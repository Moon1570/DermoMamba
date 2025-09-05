"""
Ultra-fast approximation of Mamba for DermoMamba
This maintains architectural properties but uses efficient approximations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastMambaApproximation(nn.Module):
    """
    Very fast approximation of Mamba that maintains the key properties:
    1. Sequential processing capability
    2. Gating mechanism  
    3. Multi-scale receptive field
    4. Skip connections
    
    But uses much simpler operations for speed
    """
    def __init__(self, d_model, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Simplified "state space" - just a 1D convolution for temporal mixing
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=3, padding=1, groups=self.d_inner, bias=False
        )
        
        # Activation
        self.act = nn.SiLU()
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        x: (B, L, D) or (B, H, W, D)
        """
        original_shape = x.shape
        
        # Reshape to (B, L, D) if needed
        if x.dim() == 4:
            B, H, W, D = x.shape
            x = x.view(B, H * W, D)
        
        B, L, D = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Transpose for 1D convolution: (B, L, D) -> (B, D, L)
        x = x.transpose(-1, -2)
        
        # Apply 1D convolution for sequence mixing
        x = self.conv1d(x)
        
        # Transpose back: (B, D, L) -> (B, L, D)
        x = x.transpose(-1, -2)
        
        # Apply gating
        x = x * self.act(z)
        
        # Output projection
        out = self.out_proj(x)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Reshape back if needed
        if len(original_shape) == 4:
            out = out.view(B, H, W, D)
        
        return out

class FastVSSBlock(nn.Module):
    """
    Fast VSSBlock approximation
    """
    def __init__(self, hidden_dim, drop_path=0.0, **kwargs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = FastMambaApproximation(hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, H, W, C)
        return x + self.drop_path(self.self_attention(self.ln_1(x)))

class FastCrossScaleMambaBlock(nn.Module):
    """
    Fast approximation of Cross Scale Mamba Block
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0
        quarter_dim = dim // 4
        
        # Simplified depthwise convolutions
        self.dw1 = nn.Conv2d(quarter_dim, quarter_dim, 3, padding=1, groups=quarter_dim, bias=False)
        self.dw2 = nn.Conv2d(quarter_dim, quarter_dim, 3, padding=2, dilation=2, groups=quarter_dim, bias=False)
        self.dw3 = nn.Conv2d(quarter_dim, quarter_dim, 3, padding=3, dilation=3, groups=quarter_dim, bias=False)
        
        # Fast VSS block
        self.vss = FastVSSBlock(quarter_dim)
        
        # Normalization and activation
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split into 4 parts
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        
        # Process first 3 parts
        y1 = self.dw1(x1)
        y1 = y1.permute(0, 2, 3, 1)  # BCHW -> BHWC
        y1 = self.vss(y1)
        y1 = y1.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        y2 = self.dw2(x2)
        y2 = y2.permute(0, 2, 3, 1)
        y2 = self.vss(y2)
        y2 = y2.permute(0, 3, 1, 2)
        
        y3 = self.dw3(x3)
        y3 = y3.permute(0, 2, 3, 1)
        y3 = self.vss(y3)
        y3 = y3.permute(0, 3, 1, 2)
        
        # Fourth part unchanged
        y4 = x4
        
        # Concatenate
        y = torch.cat([y1, y2, y3, y4], dim=1)
        
        # Apply normalization and activation
        return self.act(self.bn(y))

class FastSweepMamba(nn.Module):
    """
    Fast approximation of Sweep Mamba
    """
    def __init__(self, dim, ratio=8):
        super().__init__()
        reduced_dim = max(dim // ratio, 16)
        
        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, reduced_dim, bias=False)
        self.proj_out = nn.Linear(reduced_dim, dim, bias=False)
        
        # Simplified processing - just depthwise convolutions in different directions
        self.conv_h = nn.Conv2d(reduced_dim, reduced_dim, (1, 7), padding=(0, 3), groups=reduced_dim, bias=False)
        self.conv_v = nn.Conv2d(reduced_dim, reduced_dim, (7, 1), padding=(3, 0), groups=reduced_dim, bias=False)
        self.conv_d = nn.Conv2d(reduced_dim, reduced_dim, 3, padding=1, groups=reduced_dim, bias=False)
        
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm2d(dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        skip = x
        
        # Convert to BHWC
        x = x.permute(0, 2, 3, 1)
        
        # Project to reduced dimension
        x = self.proj_in(self.ln(x))
        
        # Convert back to BCHW
        x = x.permute(0, 3, 1, 2)
        
        # Apply different directional convolutions
        x_h = self.conv_h(x)
        x_v = self.conv_v(x)
        x_d = self.conv_d(x)
        
        # Combine
        x = x_h + x_v + x_d
        
        # Apply activation
        x = self.act(x)
        
        # Convert back to BHWC for projection
        x = x.permute(0, 2, 3, 1)
        x = self.proj_out(x)
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection
        out = x + skip * self.scale
        
        return F.relu(self.bn(out))

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
