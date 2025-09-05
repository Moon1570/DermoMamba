"""
CPU-compatible fallback for mamba_ssm operations
This provides basic functionality for testing and development without CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    CPU fallback for selective scan function
    This is a simplified implementation for testing purposes
    """
    batch_size, seq_len, d_inner = u.shape
    
    # Simple linear transformation as fallback
    x = u
    if z is not None:
        x = x * torch.sigmoid(z)
    
    # Apply matrices
    if B is not None:
        x = torch.matmul(x, B.transpose(-2, -1))
    if C is not None:
        x = torch.matmul(x, C)
    
    if D is not None:
        x = x + D.unsqueeze(0).unsqueeze(0) * u
    
    if return_last_state:
        return x, x[:, -1:]
    return x

def selective_scan_ref(*args, **kwargs):
    """Reference implementation (same as fn for CPU)"""
    return selective_scan_fn(*args, **kwargs)

# Mock the mamba_ssm module structure
class MockMambaSSM:
    class ops:
        class selective_scan_interface:
            selective_scan_fn = selective_scan_fn
            selective_scan_ref = selective_scan_ref

# Create mock module
import sys
sys.modules['mamba_ssm'] = MockMambaSSM()
sys.modules['mamba_ssm.ops'] = MockMambaSSM.ops()
sys.modules['mamba_ssm.ops.selective_scan_interface'] = MockMambaSSM.ops.selective_scan_interface()
