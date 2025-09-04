"""
Script to monitor GPU usage during DermoMamba training
"""
import os
import torch
import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU: {nvmlDeviceGetName(handle)}")
    print(f"Memory: {info.used/1024**3:.2f} GB / {info.total/1024**3:.2f} GB ({info.used/info.total*100:.2f}%)")
    
def print_cpu_utilization():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")

def monitor(interval=2.0, duration=60):
    """Monitor GPU and CPU usage for the specified duration in seconds"""
    start_time = time.time()
    while time.time() - start_time < duration:
        print("\n" + "="*50)
        print(f"Time: {time.time() - start_time:.2f} seconds")
        try:
            print_gpu_utilization()
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        print_cpu_utilization()
        
        # Print PyTorch memory stats
        if torch.cuda.is_available():
            print(f"PyTorch allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"PyTorch reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            
            # Print per-tensor memory usage (top 5 by size)
            if hasattr(torch.cuda, 'memory_snapshot'):
                snapshot = torch.cuda.memory_snapshot()
                if 'segments' in snapshot and len(snapshot['segments']) > 0:
                    print("\nLargest memory blocks:")
                    blocks = []
                    for segment in snapshot['segments']:
                        for block in segment.get('blocks', []):
                            if block['state'] == 'active' and 'size' in block:
                                blocks.append(block)
                    
                    blocks.sort(key=lambda b: b.get('size', 0), reverse=True)
                    for i, block in enumerate(blocks[:5]):
                        print(f"  Block {i+1}: {block.get('size', 0)/1024**2:.2f} MB")
        
        time.sleep(interval)

if __name__ == "__main__":
    try:
        import pynvml
        import psutil
    except ImportError:
        print("Installing required packages...")
        os.system("pip install pynvml psutil")
        import pynvml
        import psutil
    
    print("Starting GPU monitoring...")
    print("(Press Ctrl+C to stop)")
    try:
        monitor(interval=2.0, duration=3600)  # Monitor for 1 hour max
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except Exception as e:
        print(f"\nError: {e}")
