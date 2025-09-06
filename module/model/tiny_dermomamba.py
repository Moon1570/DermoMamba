"""
Memory-efficient ultra-fast DermoMamba implementation
Designed for 8GB GPU with minimal memory footprint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryEfficientMamba(nn.Module):
    """
    Ultra-lightweight Mamba approximation with minimal memory usage
    """
    def __init__(self, d_model, expand=1.5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Very lightweight operations
        self.proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.gate = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, C)
        original_shape = x.shape
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        
        # Simple gated linear transformation
        proj_x = self.proj(x)
        gate_x = torch.sigmoid(self.gate(x))
        x = proj_x * gate_x
        x = self.act(x)
        x = self.out_proj(x)
        
        return x.view(original_shape)

class LightweightBlock(nn.Module):
    """
    Ultra-lightweight block to replace heavy Mamba blocks
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)  # More memory efficient than LayerNorm
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pwconv = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.pwconv(x)
        x = self.act(x)
        return x + skip

class TinyDermoMamba(nn.Module):
    """
    Memory-efficient tiny DermoMamba for 8GB GPU
    """
    def __init__(self, n_class=1):
        super().__init__()
        
        # Much smaller channel dimensions
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Encoder with minimal channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            LightweightBlock(64)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            LightweightBlock(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            LightweightBlock(256)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1, bias=False),  # 256 + 128 from skip = 384
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1, bias=False),   # 128 + 64 from skip = 192
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1, bias=False),    # 64 + 32 from skip = 96
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.Conv2d(16, n_class, 1)
        )
        
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
        # Input: (B, 3, H, W)
        
        # Stem
        x = self.stem(x)  # (B, 32, H/4, W/4)
        stem_skip = x
        
        # Encoder
        skip1 = self.enc1(x)      # (B, 64, H/4, W/4)
        skip2 = self.enc2(skip1)  # (B, 128, H/8, W/8)
        skip3 = self.enc3(skip2)  # (B, 256, H/16, W/16)
        
        # Bottleneck
        x = self.bottleneck(skip3)  # (B, 256, H/32, W/32)
        
        # Decoder
        x = self.up3(x)  # (B, 128, H/16, W/16)
        # Resize skip3 if needed
        if x.size()[2:] != skip3.size()[2:]:
            skip3 = F.interpolate(skip3, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)  # (B, 128+256=384, H/16, W/16)
        x = self.dec3(x)  # (B, 128, H/16, W/16)
        
        x = self.up2(x)  # (B, 64, H/8, W/8)
        # Resize skip2 if needed
        if x.size()[2:] != skip2.size()[2:]:
            skip2 = F.interpolate(skip2, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)  # (B, 64+128=192, H/8, W/8)
        x = self.dec2(x)  # (B, 64, H/8, W/8)
        
        x = self.up1(x)  # (B, 32, H/4, W/4)
        # Resize skip1 if needed
        if x.size()[2:] != skip1.size()[2:]:
            skip1 = F.interpolate(skip1, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)  # (B, 32+64=96, H/4, W/4)
        x = self.dec1(x)  # (B, 32, H/4, W/4)
        
        # Final output
        x = self.final(x)  # (B, n_class, H, W)
        
        return x

# For debugging/testing
if __name__ == "__main__":
    model = TinyDermoMamba(n_class=1)
    
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
