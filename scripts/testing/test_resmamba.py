"""
Test TinyDermoMamba with ResMambaBlock
"""
import torch
import sys
import os
sys.path.append('.')

def test_resmamba_model():
    print("="*60)
    print("Testing TinyDermoMamba with ResMambaBlock")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {gpu_memory:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial memory: {initial_memory:.1f} MB")
    
    try:
        from module.model.tiny_dermomamba_resmamba import TinyDermoMambaWithResMamba
        
        # Create model
        model = TinyDermoMambaWithResMamba(n_class=1).to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        if device.type == 'cuda':
            model_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Memory after model load: {model_memory:.1f} MB")
        
        # Test different input sizes
        test_sizes = [
            (1, 256, 256),
            (2, 256, 256),
            (1, 384, 384),
            (2, 192, 192),
        ]
        
        for batch_size, h, w in test_sizes:
            try:
                print(f"\nTesting batch_size={batch_size}, size={h}x{w}")
                
                x = torch.randn(batch_size, 3, h, w, device=device)
                
                # Clear cache before test
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    pre_memory = torch.cuda.memory_allocated() / 1024**2
                
                # Warmup
                with torch.no_grad():
                    for _ in range(2):
                        _ = model(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Timing test
                import time
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(3):
                        output = model(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                avg_time = (time.time() - start_time) / 3
                
                if device.type == 'cuda':
                    peak_memory = torch.cuda.memory_allocated() / 1024**2
                    memory_used = peak_memory - pre_memory
                    print(f"  Time: {avg_time:.3f}s, Memory: {memory_used:.1f}MB, Output: {output.shape}")
                else:
                    print(f"  Time: {avg_time:.3f}s, Output: {output.shape}")
                
                # Clean up
                del x, output
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ Out of memory for batch_size={batch_size}, size={h}x{w}")
                else:
                    print(f"  ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Clean up after error
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        print("\n" + "="*60)
        print("✅ TinyDermoMamba with ResMambaBlock test completed!")
        
        # Compare with original tiny model
        print("\nComparing with original TinyDermoMamba...")
        try:
            from module.model.tiny_dermomamba import TinyDermoMamba
            original_model = TinyDermoMamba(n_class=1)
            original_params = sum(p.numel() for p in original_model.parameters())
            
            print(f"Original TinyDermoMamba: {original_params:,} parameters")
            print(f"ResMamba TinyDermoMamba: {total_params:,} parameters")
            print(f"Parameter difference: {total_params - original_params:,} ({(total_params/original_params):.2f}x)")
            
        except Exception as e:
            print(f"Could not compare with original: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_resmamba_model()
