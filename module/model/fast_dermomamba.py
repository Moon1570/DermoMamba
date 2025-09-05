"""
Ultra-fast DermoMamba implementation
Uses efficient approximations while maintaining architectural properties
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.fast_mamba import FastCrossScaleMambaBlock, FastSweepMamba

class FastEncoderBlock(nn.Module):
    """Very fast encoder block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Simple efficient convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Fast cross-scale Mamba block
        self.mamba = FastCrossScaleMambaBlock(out_ch)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.mamba(x)
        return x

class FastDecoderBlock(nn.Module):
    """Very fast decoder block"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Resize skip connection to match x if needed
        if x.size()[2:] != skip.size()[2:]:
            skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FastDermoMamba(nn.Module):
    """
    Ultra-fast DermoMamba implementation
    Maintains the architectural properties but uses efficient approximations
    """
    def __init__(self, n_class=1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Encoder blocks with much faster operations
        self.encoder1 = FastEncoderBlock(64, 128)
        self.encoder2 = FastEncoderBlock(128, 256)
        self.encoder3 = FastEncoderBlock(256, 512)
        
        # Bottleneck - simplified
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            FastSweepMamba(1024, ratio=4)  # Less reduction for speed
        )
        
        # Decoder blocks
        self.decoder3 = FastDecoderBlock(1024, 512, 512)
        self.decoder2 = FastDecoderBlock(512, 256, 256)
        self.decoder1 = FastDecoderBlock(256, 128, 128)
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Input: (B, 3, H, W)
        
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x
        x = self.maxpool(x)
        
        # Encoder
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        
        # Bottleneck
        x = self.bottleneck(skip3)
        
        # Decoder
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        
        # Resize to match input size if needed
        if x.size()[2:] != skip0.size()[2:]:
            x = F.interpolate(x, size=skip0.size()[2:], mode='bilinear', align_corners=False)
        
        return x

# For debugging/testing
if __name__ == "__main__":
    model = FastDermoMamba(n_class=1)
    
    # Test with dummy input
    x = torch.randn(2, 3, 256, 256)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
