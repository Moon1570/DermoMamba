"""
Spatial-Aware DermoMamba Model
Modified to handle 6-channel input: RGB + X + Y + Radial coordinates
"""

import torch
import torch.nn as nn
from module.model.optimized_dermomamba_complete import OptimizedDermoMamba

class SpatialAwareDermoMamba(OptimizedDermoMamba):
    """
    Extended DermoMamba that handles 6-channel input:
    - Channels 0-2: RGB image
    - Channel 3: X coordinates 
    - Channel 4: Y coordinates
    - Channel 5: Radial distance from center
    """
    
    def __init__(self, n_class=1, input_channels=6):
        super().__init__(n_class=n_class)
        
        # Replace the initial projection to handle 6 channels
        self.pw_in = nn.Conv2d(input_channels, 16, 1, bias=False)
        
        # Add spatial feature fusion layer
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 16, 1, bias=False)
        )
    
    def forward(self, x):
        # x shape: (B, 6, H, W) - RGB + spatial coordinates
        
        # Initial projection with 6-channel input
        x = self.pw_in(x)  # (B, 16, H, W)
        
        # Apply spatial feature fusion
        x = self.spatial_fusion(x)  # (B, 16, H, W)
        
        # Continue with the original DermoMamba forward pass
        # Encoder path - handle tuple returns (downsampled, skip)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)
        
        # Process skip connections with CBAM (as in paper)
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)
        
        # Bottleneck processing (as in paper)
        x = self.b1(x)  # Sweep_Mamba
        x = self.b2(x)  # PCA attention
        
        # Decoder path with skip connections (as in paper)
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        
        # Final output
        x = self.conv_out(x)
        
        return x
