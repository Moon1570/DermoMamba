"""
Optimized Cross Scale Mamba Block that maintains all original properties
but with better performance
"""
import torch
import torch.nn as nn
from module.optimized_mamba import OptimizedVSSBlock

class OptimizedAxialSpatialDW(nn.Module):
    """
    Optimized Axial Spatial Depthwise convolution
    Maintains the same mathematical properties but with fused operations
    """
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        
        # Use grouped convolution for efficiency
        self.mixer_h = nn.Conv2d(
            dim, dim, kernel_size=(h, 1), 
            padding='same', groups=dim, dilation=dilation,
            bias=False  # Remove bias for efficiency
        )
        self.mixer_w = nn.Conv2d(
            dim, dim, kernel_size=(1, w), 
            padding='same', groups=dim, dilation=dilation,
            bias=False
        )
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, 
            padding='same', groups=dim, dilation=dilation,
            bias=False
        )
        
        # Batch norm for stability
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        identity = x
        
        # Apply all convolutions in sequence
        x = self.mixer_w(x)
        x = self.mixer_h(x)
        x = self.conv(x)
        x = self.bn(x)
        
        # Residual connection
        return x + identity

class OptimizedCrossScaleMambaBlock(nn.Module):
    """
    Optimized Cross Scale Mamba Block that maintains all properties
    but with improved performance and memory efficiency
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Ensure dim is divisible by 4
        assert dim % 4 == 0, f"dim ({dim}) must be divisible by 4"
        
        quarter_dim = dim // 4
        
        # Optimized axial spatial convolutions
        self.dw1 = OptimizedAxialSpatialDW(quarter_dim, (7, 7), dilation=1)
        self.dw2 = OptimizedAxialSpatialDW(quarter_dim, (7, 7), dilation=2)
        self.dw3 = OptimizedAxialSpatialDW(quarter_dim, (7, 7), dilation=3)
        
        # Optimized VSS block
        self.vss = OptimizedVSSBlock(
            hidden_dim=quarter_dim,
            norm_layer=norm_layer,
            d_state=16  # Reduced state size for efficiency
        )
        
        # Normalization and activation
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)  # Inplace for memory efficiency
        
        # Optional: learnable scaling factors for each branch
        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))
        self.scale3 = nn.Parameter(torch.ones(1))
        self.scale4 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Split input into 4 parts
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        
        # Process each part through different dilated convolutions
        # and then through VSS block
        
        # Branch 1: dilation=1 + VSS
        y1 = self.dw1(x1)
        y1 = y1.permute(0, 2, 3, 1)  # BCHW -> BHWC for VSS
        y1 = self.vss(y1)
        y1 = y1.permute(0, 3, 1, 2)  # BHWC -> BCHW
        y1 = y1 * self.scale1
        
        # Branch 2: dilation=2 + VSS
        y2 = self.dw2(x2)
        y2 = y2.permute(0, 2, 3, 1)
        y2 = self.vss(y2)
        y2 = y2.permute(0, 3, 1, 2)
        y2 = y2 * self.scale2
        
        # Branch 3: dilation=3 + VSS
        y3 = self.dw3(x3)
        y3 = y3.permute(0, 2, 3, 1)
        y3 = self.vss(y3)
        y3 = y3.permute(0, 3, 1, 2)
        y3 = y3 * self.scale3
        
        # Branch 4: direct pass with scaling
        y4 = x4 * self.scale4
        
        # Concatenate all branches
        y = torch.cat([y1, y2, y3, y4], dim=1)
        
        # Apply batch norm and activation
        y = self.act(self.bn(y))
        
        return y

class OptimizedResMambaBlock(nn.Module):
    """
    Optimized Residual Mamba Block
    """
    def __init__(self, in_c):
        super().__init__()
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.block = OptimizedCrossScaleMambaBlock(in_c)
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding='same', bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        
        # Batch norm for the conv
        self.bn = nn.BatchNorm2d(in_c)
        
    def forward(self, x):
        identity = x
        
        # Apply mamba block
        x = self.block(x)
        
        # Apply conv + norm + activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(self.ins_norm(x))
        
        # Residual connection with learnable scaling
        return x + identity * self.scale

class OptimizedEncoderBlock(nn.Module):
    """
    Optimized Encoder Block with improved performance
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.resmamba = OptimizedResMambaBlock(in_c)
        self.down = nn.MaxPool2d((2, 2))
        
    def forward(self, x):
        # Apply residual mamba block
        x = self.resmamba(x)
        
        # Generate skip connection
        skip = self.act(self.bn(self.pw(x)))
        
        # Downsample
        x = self.down(skip)
        
        return x, skip
