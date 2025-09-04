import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capabilities: {torch.cuda.get_device_capability(0)}")
    # Try a simple CUDA operation to verify it works
    x = torch.rand(3, 3).cuda()
    print(f"Tensor on device: {x.device}")
else:
    print("No CUDA devices available")

# Check if CUDA drivers are compatible with this PyTorch build
if hasattr(torch, '_C'):
    if hasattr(torch._C, '_cuda_getDriverVersion'):
        print(f"CUDA driver version: {torch._C._cuda_getDriverVersion() / 1000:.1f}")
    if hasattr(torch._C, '_cuda_getRuntimeVersion'):
        print(f"CUDA runtime version: {torch._C._cuda_getRuntimeVersion() / 1000:.1f}")

# Memory information
if torch.cuda.is_available():
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Reserved GPU memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
