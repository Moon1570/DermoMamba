"""
Optimized DermoMamba model that maintains all original properties
but with significant performance improvements
"""
import torch
import torch.nn as nn
from module.optimized_blocks import OptimizedEncoderBlock
from module.model.decoder import DecoderBlock
from module.CBAM import CBAM
from module.optimized_bottleneck import OptimizedSweepMamba, OptimizedPCA

class OptimizedDermoMamba(nn.Module):
    """
    Optimized version of DermoMamba that maintains all architectural properties
    but with performance optimizations:
    1. Optimized encoder blocks with faster Mamba implementations
    2. Gradient checkpointing for memory efficiency
    3. Fused operations where possible
    4. Better memory management
    """
    def __init__(self, use_gradient_checkpointing=True):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.act_in = nn.ReLU(inplace=True)
        
        # Optimized encoder blocks
        self.e1 = OptimizedEncoderBlock(16, 32)
        self.e2 = OptimizedEncoderBlock(32, 64)
        self.e3 = OptimizedEncoderBlock(64, 128)
        self.e4 = OptimizedEncoderBlock(128, 256)
        self.e5 = OptimizedEncoderBlock(256, 512)
        
        # Skip connection attention (kept original for compatibility)
        self.s1 = CBAM(32)
        self.s2 = CBAM(64)
        self.s3 = CBAM(128)
        self.s4 = CBAM(256)
        self.s5 = CBAM(512)
        
        # Optimized bottleneck components
        self.b1 = OptimizedSweepMamba(512)
        self.b2 = OptimizedPCA(512)
        
        # Decoder blocks (kept original)
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        
        # Final output
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
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
        # Input projection with activation
        x = self.act_in(self.bn_in(self.pw_in(x)))
        
        # Encoder path with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            x, skip1 = torch.utils.checkpoint.checkpoint(self.e1, x)
            x, skip2 = torch.utils.checkpoint.checkpoint(self.e2, x)
            x, skip3 = torch.utils.checkpoint.checkpoint(self.e3, x)
            x, skip4 = torch.utils.checkpoint.checkpoint(self.e4, x)
            x, skip5 = torch.utils.checkpoint.checkpoint(self.e5, x)
        else:
            x, skip1 = self.e1(x)
            x, skip2 = self.e2(x)
            x, skip3 = self.e3(x)
            x, skip4 = self.e4(x)
            x, skip5 = self.e5(x)
        
        # Skip connection processing
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)
        
        # Bottleneck with residual connection
        bottleneck_out = self.b2(x)
        x = self.b1(bottleneck_out + x)
        
        # Decoder path
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        
        # Final output
        x = self.conv_out(x)
        
        return x
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.use_gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for faster inference"""
        self.use_gradient_checkpointing = False

# Alias for backward compatibility
OptimizedDermoMamba = OptimizedDermoMamba
