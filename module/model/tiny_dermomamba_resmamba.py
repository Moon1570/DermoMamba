"""
TinyDermoMamba with proper ResMambaBlock architecture
Memory-efficient version that maintains the original DermoMamba structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.CSMB import Cross_Scale_Mamba_Block

class TinyResMambaBlock(nn.Module):
    """
    Memory-efficient version of ResMambaBlock using GroupNorm and smaller channels
    """
    def __init__(self, in_c):
        super().__init__()
        # Use GroupNorm instead of InstanceNorm for memory efficiency
        self.norm = nn.GroupNorm(min(8, in_c), in_c, affine=True)  
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.block = Cross_Scale_Mamba_Block(in_c)
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Apply Cross_Scale_Mamba_Block first
        mamba_out = self.block(x)
        # Apply conv, norm, activation and residual connection
        conv_out = self.act(self.norm(self.conv(mamba_out)))
        return conv_out + x * self.scale

class TinyEncoderBlock(nn.Module):
    """
    Memory-efficient EncoderBlock using ResMambaBlock
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Project to output channels first
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.GroupNorm(min(8, out_c), out_c)  # Use GroupNorm for efficiency
        self.act = nn.ReLU(inplace=True)
        self.resmamba = TinyResMambaBlock(in_c)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Apply ResMambaBlock first
        x = self.resmamba(x)
        # Then project and get skip connection
        skip = self.act(self.bn(self.pw(x)))
        # Downsample for next level
        x = self.down(skip)
        return x, skip

class TinyDermoMambaWithResMamba(nn.Module):
    """
    Memory-efficient DermoMamba that properly uses ResMambaBlock like the original
    but with smaller channel dimensions for 8GB GPU
    """
    def __init__(self, n_class=1):
        super().__init__()
        
        # Initial projection (same as original but smaller channels)
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        
        # Encoder blocks using ResMambaBlock (smaller channels)
        self.e1 = TinyEncoderBlock(16, 32)
        self.e2 = TinyEncoderBlock(32, 64)
        self.e3 = TinyEncoderBlock(64, 128)
        self.e4 = TinyEncoderBlock(128, 256)
        
        # Skip connection processing (simplified CBAM)
        self.s1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 1, bias=False),
            nn.Sigmoid()
        )
        self.s2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 1, bias=False),
            nn.Sigmoid()
        )
        self.s3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1, bias=False),
            nn.Sigmoid()
        )
        self.s4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Bottleneck (simplified versions of Sweep_Mamba and PCA)
        self.b1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Decoder blocks (fixed channel dimensions)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(384, 64, 2, stride=2, bias=False),  # 256 + 128 from skip = 384
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(192, 32, 2, stride=2, bias=False),  # 128 + 64 from skip = 192
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(96, 16, 2, stride=2, bias=False),   # 64 + 32 from skip = 96
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )
        
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
        
        # Encoder path with ResMambaBlocks
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        
        # Apply skip connection processing
        skip1 = skip1 * self.s1(skip1)
        skip2 = skip2 * self.s2(skip2)
        skip3 = skip3 * self.s3(skip3)
        skip4 = skip4 * self.s4(skip4)
        
        # Bottleneck
        bottleneck_out = self.b2(x)
        x = self.b1(bottleneck_out * x + x)
        
        # Decoder path
        x = self.d4(x)
        # Concatenate with skip4
        if x.size()[2:] != skip4.size()[2:]:
            skip4 = F.interpolate(skip4, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        
        x = self.d3(x)
        # Concatenate with skip3
        if x.size()[2:] != skip3.size()[2:]:
            skip3 = F.interpolate(skip3, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        
        x = self.d2(x)
        # Concatenate with skip2
        if x.size()[2:] != skip2.size()[2:]:
            skip2 = F.interpolate(skip2, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        
        x = self.d1(x)
        # Resize to match original input size if needed
        target_size = (x.size(2) * 2, x.size(3) * 2)  # Upsample by 2x
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Final output
        x = self.conv_out(x)
        
        return x

# For debugging/testing
if __name__ == "__main__":
    model = TinyDermoMambaWithResMamba(n_class=1)
    
    # Test with dummy input
    x = torch.randn(1, 3, 256, 256)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
