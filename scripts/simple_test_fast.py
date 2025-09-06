"""
Simple test for fast DermoMamba
"""
import torch
import time
import sys
import os
sys.path.append('.')

def main():
    print("Testing Fast DermoMamba...")
    
    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("❌ Using CPU")
    
    try:
        from module.model.fast_dermomamba import FastDermoMamba
        print("✅ Successfully imported FastDermoMamba")
        
        # Create model
        model = FastDermoMamba(n_class=1)
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model parameters: {total_params:,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224, device=device)
        
        print("Testing forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            output = model(x)
        
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Forward pass time: {forward_time:.3f}s")
        
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"✅ GPU Memory: {memory_allocated:.2f} MB")
        
        # Test with batch
        print("\nTesting with batch size 2...")
        x_batch = torch.randn(2, 3, 224, 224, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            output_batch = model(x_batch)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch forward pass time: {batch_time:.3f}s")
        print(f"✅ Time per sample: {batch_time/2:.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
