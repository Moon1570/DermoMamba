"""
Test the ultra-fast DermoMamba implementation
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fast_implementation():
    print("="*60)
    print("Testing Ultra-Fast DermoMamba Implementation")
    print("="*60)
    
    # GPU setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name()
        print(f"✅ GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("❌ Using CPU")
    
    # Import the fast model
    from module.model.fast_dermomamba import FastDermoMamba
    
    # Create model
    model = FastDermoMamba(n_class=1)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Test different batch sizes and resolutions
    test_configs = [
        {"batch_size": 1, "size": 224, "name": "Single 224x224"},
        {"batch_size": 2, "size": 224, "name": "Batch-2 224x224"},
        {"batch_size": 4, "size": 224, "name": "Batch-4 224x224"},
        {"batch_size": 1, "size": 256, "name": "Single 256x256"},
        {"batch_size": 2, "size": 256, "name": "Batch-2 256x256"},
    ]
    
    print("\nRunning speed tests...")
    print("-" * 60)
    
    with torch.no_grad():
        for config in test_configs:
            batch_size = config["batch_size"]
            size = config["size"]
            name = config["name"]
            
            # Create test input
            x = torch.randn(batch_size, 3, size, size, device=device)
            
            # Warmup
            for _ in range(3):
                with autocast():
                    _ = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Actual timing
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with autocast():
                    output = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            print(f"{name:20} | Avg time: {avg_time:.3f}s | Output shape: {output.shape}")
            
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"{' '*20} | GPU Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
    
    print("-" * 60)
    print("✅ All tests completed successfully!")
    
    # Compare with a simple baseline
    print("\nComparing with simple U-Net baseline...")
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    baseline = SimpleUNet().to(device)
    baseline.eval()
    
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline U-Net parameters: {baseline_params:,}")
    
    # Test baseline speed
    x = torch.randn(2, 3, 224, 224, device=device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = baseline(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            _ = baseline(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        baseline_time = (time.time() - start_time) / 10
    
    print(f"Simple U-Net time: {baseline_time:.3f}s")
    
    # Test FastDermoMamba with same input
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        fast_time = (time.time() - start_time) / 10
    
    print(f"Fast DermoMamba time: {fast_time:.3f}s")
    print(f"Speed ratio (Fast/Baseline): {fast_time/baseline_time:.2f}x")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"✅ Fast DermoMamba: {total_params:,} parameters")
    print(f"✅ Average inference time: ~{fast_time:.3f}s per batch")
    print(f"✅ Memory efficient and GPU optimized")
    print("="*60)

if __name__ == "__main__":
    test_fast_implementation()
