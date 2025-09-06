"""
GPU Utilities for DermoMamba

Functions for GPU monitoring, status checking, and system verification.
"""

import torch
import psutil
import subprocess
import time
from typing import Dict, List, Tuple, Optional

def check_gpu_status() -> Dict:
    """
    Check GPU status and availability
    
    Returns:
        dict: GPU status information including device count, memory, etc.
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'device_name': None,
        'memory_total': 0,
        'memory_free': 0,
        'memory_used': 0
    }
    
    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()
        gpu_info['current_device'] = torch.cuda.current_device()
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        
        # Memory information
        memory_stats = torch.cuda.memory_stats()
        gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        gpu_info['memory_free'] = gpu_info['memory_total'] - torch.cuda.memory_allocated()
        gpu_info['memory_used'] = torch.cuda.memory_allocated()
    
    return gpu_info

def monitor_gpu_memory(interval: int = 1, duration: int = 60) -> List[Dict]:
    """
    Monitor GPU memory usage over time
    
    Args:
        interval: Monitoring interval in seconds
        duration: Total monitoring duration in seconds
    
    Returns:
        list: Memory usage data over time
    """
    if not torch.cuda.is_available():
        return []
    
    memory_log = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        memory_info = {
            'timestamp': time.time(),
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
        }
        memory_log.append(memory_info)
        time.sleep(interval)
    
    return memory_log

def print_gpu_status():
    """Print formatted GPU status information"""
    gpu_info = check_gpu_status()
    
    print("=" * 50)
    print("🔧 GPU STATUS")
    print("=" * 50)
    print(f"CUDA Available: {'✅ Yes' if gpu_info['cuda_available'] else '❌ No'}")
    
    if gpu_info['cuda_available']:
        print(f"Device Count: {gpu_info['device_count']}")
        print(f"Current Device: {gpu_info['current_device']}")
        print(f"Device Name: {gpu_info['device_name']}")
        print(f"Total Memory: {gpu_info['memory_total'] / (1024**3):.2f} GB")
        print(f"Used Memory: {gpu_info['memory_used'] / (1024**3):.2f} GB")
        print(f"Free Memory: {gpu_info['memory_free'] / (1024**3):.2f} GB")
    
    print("=" * 50)

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cache cleared")
    else:
        print("❌ CUDA not available")

def get_system_info() -> Dict:
    """Get system information including CPU, RAM, etc."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent if psutil.disk_usage('/') else None
    }

if __name__ == "__main__":
    print_gpu_status()
    system_info = get_system_info()
    print(f"\n💻 System Info:")
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"RAM Total: {system_info['memory_total'] / (1024**3):.2f} GB")
    print(f"RAM Available: {system_info['memory_available'] / (1024**3):.2f} GB")
