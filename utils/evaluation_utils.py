"""
Evaluation Utilities for DermoMamba

Functions for model evaluation, comparison, and result extraction.
"""

import os
import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pytorch_lightning as pl

def extract_model_results(checkpoint_dir: str, model_name: str = None) -> Dict[str, Any]:
    """
    Extract training results from PyTorch Lightning checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_name: Optional model name for filtering
    
    Returns:
        dict: Extracted results with metrics and metadata
    """
    results = {
        'model_name': model_name or 'Unknown',
        'checkpoints': [],
        'best_metrics': {},
        'training_history': []
    }
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return results
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        print(f"⚠️ No checkpoint files found in {checkpoint_dir}")
        return results
    
    best_dice = 0
    best_checkpoint = None
    
    for ckpt_file in checkpoint_files:
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            
            # Extract metrics
            ckpt_info = {
                'file': os.path.basename(ckpt_file),
                'epoch': checkpoint.get('epoch', -1),
                'global_step': checkpoint.get('global_step', -1),
                'metrics': {}
            }
            
            # Get validation metrics
            if 'state_dict' in checkpoint:
                # Try to extract metrics from different possible locations
                for key in ['val_dice', 'val_dice_score', 'dice_score', 'best_val_dice']:
                    if key in checkpoint:
                        ckpt_info['metrics']['dice'] = float(checkpoint[key])
                        break
                
                # Check callbacks state for metrics
                if 'lr_schedulers' in checkpoint:
                    for scheduler in checkpoint['lr_schedulers']:
                        if isinstance(scheduler, dict) and 'best' in scheduler:
                            ckpt_info['metrics']['dice'] = float(scheduler['best'])
            
            # Try to get metrics from the checkpoint's state
            if 'callbacks' in checkpoint:
                for callback_name, callback_state in checkpoint['callbacks'].items():
                    if 'best_model_score' in callback_state:
                        ckpt_info['metrics']['dice'] = float(callback_state['best_model_score'])
                    elif 'best_model_path' in callback_state:
                        # Extract score from filename if possible
                        path = callback_state['best_model_path']
                        if 'dice' in path:
                            try:
                                import re
                                scores = re.findall(r'dice[_-]?(\d+\.?\d*)', path)
                                if scores:
                                    ckpt_info['metrics']['dice'] = float(scores[0])
                            except:
                                pass
            
            results['checkpoints'].append(ckpt_info)
            
            # Track best performance
            current_dice = ckpt_info['metrics'].get('dice', 0)
            if current_dice > best_dice:
                best_dice = current_dice
                best_checkpoint = ckpt_file
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint {ckpt_file}: {e}")
    
    # Set best metrics
    if best_dice > 0:
        results['best_metrics'] = {
            'dice_score': best_dice,
            'best_checkpoint': os.path.basename(best_checkpoint) if best_checkpoint else None
        }
    
    return results

def compare_model_performance(models_info: List[Dict]) -> Dict[str, Any]:
    """
    Compare performance across multiple models
    
    Args:
        models_info: List of model information dictionaries
    
    Returns:
        dict: Comparison results and rankings
    """
    comparison = {
        'models': models_info,
        'rankings': {
            'by_accuracy': [],
            'by_speed': [],
            'by_efficiency': []
        },
        'summary': {}
    }
    
    # Rank by accuracy (dice score)
    models_with_dice = [m for m in models_info if 'dice_score' in m and m['dice_score'] > 0]
    comparison['rankings']['by_accuracy'] = sorted(
        models_with_dice, 
        key=lambda x: x['dice_score'], 
        reverse=True
    )
    
    # Rank by speed (inference time)
    models_with_speed = [m for m in models_info if 'inference_time' in m and m['inference_time'] > 0]
    comparison['rankings']['by_speed'] = sorted(
        models_with_speed,
        key=lambda x: x['inference_time']
    )
    
    # Calculate efficiency score (accuracy / inference_time)
    models_with_both = [m for m in models_info if 'dice_score' in m and 'inference_time' in m 
                       and m['dice_score'] > 0 and m['inference_time'] > 0]
    
    for model in models_with_both:
        model['efficiency_score'] = model['dice_score'] / model['inference_time']
    
    comparison['rankings']['by_efficiency'] = sorted(
        models_with_both,
        key=lambda x: x.get('efficiency_score', 0),
        reverse=True
    )
    
    # Generate summary
    if comparison['rankings']['by_accuracy']:
        best_accuracy = comparison['rankings']['by_accuracy'][0]
        comparison['summary']['best_accuracy'] = {
            'model': best_accuracy.get('name', 'Unknown'),
            'dice_score': best_accuracy['dice_score']
        }
    
    if comparison['rankings']['by_speed']:
        fastest = comparison['rankings']['by_speed'][0]
        comparison['summary']['fastest'] = {
            'model': fastest.get('name', 'Unknown'),
            'inference_time': fastest['inference_time']
        }
    
    if comparison['rankings']['by_efficiency']:
        most_efficient = comparison['rankings']['by_efficiency'][0]
        comparison['summary']['most_efficient'] = {
            'model': most_efficient.get('name', 'Unknown'),
            'efficiency_score': most_efficient['efficiency_score']
        }
    
    return comparison

def calculate_model_metrics(model: torch.nn.Module, input_size: Tuple = (1, 3, 256, 256)) -> Dict[str, Any]:
    """
    Calculate comprehensive model metrics
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for calculations
    
    Returns:
        dict: Model metrics including parameters, size, etc.
    """
    metrics = {
        'parameters': 0,
        'trainable_parameters': 0,
        'model_size_mb': 0,
        'input_size': input_size
    }
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics['parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        metrics['model_size_mb'] = model_size_mb
        
    except Exception as e:
        print(f"⚠️ Error calculating model metrics: {e}")
    
    return metrics

def benchmark_inference_speed(
    model: torch.nn.Module, 
    input_size: Tuple = (1, 3, 256, 256),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run on
    
    Returns:
        dict: Timing metrics
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    return {
        'total_time': total_time,
        'average_time': avg_time,
        'throughput': 1.0 / avg_time,
        'num_runs': num_runs,
        'device': str(device)
    }

def save_evaluation_results(results: Dict, output_file: str):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def load_evaluation_results(input_file: str) -> Dict:
    """
    Load evaluation results from JSON file
    
    Args:
        input_file: Input file path
    
    Returns:
        dict: Loaded results
    """
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Results loaded from {input_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return {}

if __name__ == "__main__":
    print("🔧 Evaluation Utilities Demo")
    
    # Example: Extract results from checkpoints
    checkpoint_dirs = [
        "checkpoints/optimized_complete",
        "checkpoints/optimized_complete_improved",
        "checkpoints/tiny"
    ]
    
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            results = extract_model_results(ckpt_dir, os.path.basename(ckpt_dir))
            print(f"\n📊 Results for {ckpt_dir}:")
            print(f"Best Dice: {results['best_metrics'].get('dice_score', 'N/A')}")
            print(f"Checkpoints: {len(results['checkpoints'])}")
        else:
            print(f"⚠️ Directory not found: {ckpt_dir}")
