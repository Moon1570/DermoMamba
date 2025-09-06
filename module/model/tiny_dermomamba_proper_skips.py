"""
TinyDermoMamba with proper skip connections as in the original paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.CSMB import Cross_Scale_Mamba_Block

class TinyResMambaBlock(nn.Module):
    """Memory-efficient ResMambaBlock"""
    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, in_c), in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.block = Cross_Scale_Mamba_Block(in_c)
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        mamba_out = self.block(x)
        conv_out = self.act(self.norm(self.conv(mamba_out)))
        return conv_out + x * self.scale

class TinyEncoderBlock(nn.Module):
    """EncoderBlock following original structure"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.ReLU(inplace=True)
        self.resmamba = TinyResMambaBlock(in_c)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.resmamba(x)
        skip = self.act(self.bn(self.pw(x)))
        x = self.down(skip)
        return x, skip

class TinyCBAM(nn.Module):
    """
    Memory-efficient CBAM with both channel and spatial attention
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
        )
        
    def forward(self, x):
        # Channel attention
        avg_pool = self.channel_att(x)
        max_pool = self.channel_att(F.adaptive_max_pool2d(x, 1))
        channel_att = torch.sigmoid(avg_pool + max_pool)
        x = x * channel_att
        
        # Spatial attention  
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.spatial_att(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x

class TinyDecoderBlock(nn.Module):
    """
    Proper DecoderBlock following original paper structure:
    1. Upsample input 2x
    2. Concatenate with skip connection
    3. 1x1 conv to reduce channels
    4. 3x3 conv for processing
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.pw = nn.Conv2d(in_c * 2, in_c, kernel_size=1, bias=False)  # Reduce concatenated channels
        self.bn = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skip):
        # Upsample input
        x = self.up(x)
        
        # Resize skip if needed to match upsampled input
        if x.size()[2:] != skip.size()[2:]:
            skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate upsampled input with skip
        x = torch.cat([x, skip], dim=1)
        
        # Reduce channels and process
        x = self.pw(x)
        x = self.bn(self.act(self.pw2(x)))
        
        return x

class TinyDermoMambaProperSkips(nn.Module):
    """
    TinyDermoMamba with proper skip connections exactly as in the original paper
    """
    def __init__(self, n_class=1):
        super().__init__()
        
        # Initial projection (same as original)
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        
        # Encoder blocks using ResMambaBlock (smaller channels)
        self.e1 = TinyEncoderBlock(16, 32)
        self.e2 = TinyEncoderBlock(32, 64) 
        self.e3 = TinyEncoderBlock(64, 128)
        self.e4 = TinyEncoderBlock(128, 256)
        
        # Skip connection processing using CBAM (as in original)
        self.s1 = TinyCBAM(32, reduction=4)
        self.s2 = TinyCBAM(64, reduction=8)
        self.s3 = TinyCBAM(128, reduction=16)
        self.s4 = TinyCBAM(256, reduction=32)
        
        # Bottleneck (simplified Sweep_Mamba + PCA pattern)
        self.b1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.GroupNorm(16, 512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU()
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 256, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Decoder blocks using proper DecoderBlock structure
        self.d4 = TinyDecoderBlock(256, 128)
        self.d3 = TinyDecoderBlock(128, 64)
        self.d2 = TinyDecoderBlock(64, 32)
        self.d1 = TinyDecoderBlock(32, 16)
        
        # Final output layer
        self.conv_out = nn.Conv2d(16, n_class, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input projection
        x = self.pw_in(x)
        
        # Encoder path - exactly like original
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        
        # Skip connection processing through CBAM - exactly like original
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        
        # Bottleneck - following original pattern: b1(b2(x) + x)
        bottleneck_att = self.b2(x)
        x = self.b1(bottleneck_att * x + x)
        
        # Decoder path using proper DecoderBlocks - exactly like original
        x = self.d4(x, skip4)  # Like original: d5(x, skip5)
        x = self.d3(x, skip3)  # Like original: d4(x1, skip4)
        x = self.d2(x, skip2)  # Like original: d3(x2, skip3)
        x = self.d1(x, skip1)  # Like original: d2(x3, skip2)
        
        # Final output
        x = self.conv_out(x)
        
        return x

# For debugging/testing
if __name__ == "__main__":
    model = TinyDermoMambaProperSkips(n_class=1)
    
    # Test with dummy input
    x = torch.randn(1, 3, 256, 256)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
