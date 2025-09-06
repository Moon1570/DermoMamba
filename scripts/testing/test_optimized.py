"""
Test script for optimized DermoMamba model
"""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import fallback first
from module.mamba_fallback import *
from module.model.optimized_dermomamba import OptimizedDermoMamba

def test_optimized_model():
    """Test the optimized DermoMamba model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create optimized model
    model = OptimizedDermoMamba(use_gradient_checkpointing=False)  # Disable for testing
    model = model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):.2e} parameters")
    
    # Test different batch sizes and resolutions
    test_configs = [
        (1, 96, 128),   # Small
        (2, 96, 128),   # Batch size 2
        (1, 192, 256),  # Full resolution
        (2, 192, 256),  # Full resolution batch 2
    ]
    
    for batch_size, height, width in test_configs:
        print(f"\nTesting batch_size={batch_size}, resolution={height}x{width}")
        
        # Create input
        input_tensor = torch.randn(batch_size, 3, height, width, device=device)
        
        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Time the forward pass
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  Forward pass: {elapsed:.2f} ms")
        print(f"  Output shape: {output.shape}")
        
        if device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(0) / 1024**2
            print(f"  GPU memory: {memory_mb:.2f} MB")

if __name__ == "__main__":
    test_optimized_model()
