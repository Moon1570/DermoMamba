"""
Debug the FastDermoMamba model dimensions
"""
import torch
import sys
import os
sys.path.append('.')

def debug_fast_model():
    print("Debugging FastDermoMamba model...")
    
    from module.model.fast_dermomamba import FastDermoMamba
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FastDermoMamba(n_class=1).to(device)
    model.eval()
    
    # Test with a simple input
    x = torch.randn(1, 3, 256, 256, device=device)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        try:
            # Initial processing
            x = model.conv1(x)
            print(f"After conv1: {x.shape}")
            
            x = model.bn1(x)
            x = model.relu(x)
            skip0 = x
            print(f"Skip0 shape: {skip0.shape}")
            
            x = model.maxpool(x)
            print(f"After maxpool: {x.shape}")
            
            # Encoder
            skip1 = model.encoder1(x)
            print(f"Skip1 shape: {skip1.shape}")
            
            skip2 = model.encoder2(skip1)
            print(f"Skip2 shape: {skip2.shape}")
            
            skip3 = model.encoder3(skip2)
            print(f"Skip3 shape: {skip3.shape}")
            
            # Bottleneck
            x = model.bottleneck(skip3)
            print(f"After bottleneck: {x.shape}")
            
            # Decoder
            print("Decoder phase:")
            print(f"decoder3 input: {x.shape}, skip3: {skip3.shape}")
            x = model.decoder3(x, skip3)
            print(f"After decoder3: {x.shape}")
            
            print(f"decoder2 input: {x.shape}, skip2: {skip2.shape}")
            x = model.decoder2(x, skip2)
            print(f"After decoder2: {x.shape}")
            
            print(f"decoder1 input: {x.shape}, skip1: {skip1.shape}")
            x = model.decoder1(x, skip1)
            print(f"After decoder1: {x.shape}")
            
            # Final output
            print(f"final_conv input: {x.shape}")
            x = model.final_conv(x)
            print(f"Final output: {x.shape}")
            
            print("✅ Model test successful!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_fast_model()
