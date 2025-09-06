"""
Debug bottlenecks in DermoMamba model
This script identifies which part of the model is causing the slowdown
"""
import torch
import time
import sys
import os
from module.model.proposed_net import DermoMamba

def measure_component_performance():
    """Measure the performance of different components of the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 192, 256, device=device)
    
    # Create model
    model = DermoMamba().to(device)
    model.eval()  # Set to evaluation mode to avoid dropout, etc.
    
    # Extract components for testing
    pw_in = model.pw_in
    encoder_blocks = [model.e1, model.e2, model.e3, model.e4, model.e5]
    skip_connections = [model.s1, model.s2, model.s3, model.s4, model.s5]
    bottle_neck = [model.b1, model.b2]
    decoder_blocks = [model.d5, model.d4, model.d3, model.d2, model.d1]
    conv_out = model.conv_out
    
    with torch.no_grad():
        # Test input projection
        print("\n===== TESTING INPUT PROJECTION =====")
        start_time = time.time()
        x = pw_in(input_tensor)
        elapsed = (time.time() - start_time) * 1000
        print(f"Input projection: {elapsed:.2f} ms")
        
        # Test encoders
        print("\n===== TESTING ENCODER BLOCKS =====")
        skip_outputs = []
        for i, encoder in enumerate(encoder_blocks):
            start_time = time.time()
            x, skip = encoder(x)
            elapsed = (time.time() - start_time) * 1000
            print(f"Encoder {i+1}: {elapsed:.2f} ms, output shape: {x.shape}")
            skip_outputs.append(skip)
        
        # Test skip connections
        print("\n===== TESTING SKIP CONNECTIONS =====")
        for i, (skip_conn, skip_tensor) in enumerate(zip(skip_connections, skip_outputs)):
            start_time = time.time()
            skip_output = skip_conn(skip_tensor)
            elapsed = (time.time() - start_time) * 1000
            print(f"Skip connection {i+1}: {elapsed:.2f} ms")
            skip_outputs[i] = skip_output
        
        # Test bottleneck
        print("\n===== TESTING BOTTLENECK =====")
        start_time = time.time()
        x = bottle_neck[0](bottle_neck[1](x) + x)
        elapsed = (time.time() - start_time) * 1000
        print(f"Bottleneck: {elapsed:.2f} ms")
        
        # Test decoders
        print("\n===== TESTING DECODER BLOCKS =====")
        decoder_inputs = [(x, skip_outputs[4]), 
                         (None, skip_outputs[3]), 
                         (None, skip_outputs[2]), 
                         (None, skip_outputs[1]), 
                         (None, skip_outputs[0])]
        
        for i, (decoder, (decoder_x, skip)) in enumerate(zip(decoder_blocks, decoder_inputs)):
            if i > 0:
                decoder_inputs[i] = (x, skip)
            
            start_time = time.time()
            x = decoder(decoder_inputs[i][0], decoder_inputs[i][1])
            elapsed = (time.time() - start_time) * 1000
            print(f"Decoder {i+1}: {elapsed:.2f} ms, output shape: {x.shape}")
        
        # Test output convolution
        print("\n===== TESTING OUTPUT CONVOLUTION =====")
        start_time = time.time()
        output = conv_out(x)
        elapsed = (time.time() - start_time) * 1000
        print(f"Output convolution: {elapsed:.2f} ms, final output shape: {output.shape}")
        
        # Test full forward pass for comparison
        print("\n===== TESTING FULL FORWARD PASS =====")
        start_time = time.time()
        output = model(input_tensor)
        elapsed = (time.time() - start_time) * 1000
        print(f"Full forward pass: {elapsed:.2f} ms, output shape: {output.shape}")
        print(f"Sum of component times may not equal full pass time due to PyTorch optimizations")

def test_simplified_mamba():
    """Test if the simplified Mamba implementation is causing the slowdown"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from module.SMB import Sweep_Mamba
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== TESTING SIMPLIFIED MAMBA BLOCK =====")
    print(f"Using device: {device}")
    
    # Test with different input sizes
    batch_sizes = [1, 2, 4]
    channels = 512  # Typical channel size in the model
    
    for batch_size in batch_sizes:
        # Create a Sweep_Mamba instance
        mamba_block = Sweep_Mamba(channels).to(device)
        
        # Create input tensor - shape [B, C, H, W]
        input_tensor = torch.randn(batch_size, channels, 12, 16, device=device)
        
        # Warm-up run
        with torch.no_grad():
            _ = mamba_block(input_tensor)
        
        # Measure performance
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = mamba_block(input_tensor)
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000
            
            print(f"Batch size {batch_size}: {elapsed:.2f} ms, output shape: {output.shape}")

if __name__ == "__main__":
    print("===== DERMOMAMBA BOTTLENECK ANALYSIS =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Measure component performance
    measure_component_performance()
    
    # Test simplified Mamba implementation
    test_simplified_mamba()
