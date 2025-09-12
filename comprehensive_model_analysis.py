"""
Comprehensive Model Comparison: FLOPs, Parameters, Training Time, Performance Metrics
Enhanced version for DermoMamba project - analyzing ALL checkpoints and experiments
"""
import os
import sys
import torch
import torch.nn as nn
import time
import numpy as np
import glob
import json
from datetime import datetime
from pathlib import Path

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  Warning: thop not available. FLOPs calculation will be skipped.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  Warning: psutil not available. Memory usage will be limited.")

# Add project root
sys.path.append('d:/Research/DermoMamba')

# Import models
try:
    from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OptimizedDermoMamba = None
    OPTIMIZED_AVAILABLE = False
    print("⚠️  Warning: OptimizedDermoMamba not available")

try:
    from module.model.tiny_dermomamba import TinyDermoMamba
    TINY_AVAILABLE = True
except ImportError:
    TinyDermoMamba = None
    TINY_AVAILABLE = False

# Import metrics
from metric.metrics import dice_score, iou_score

class ModelProfiler:
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        if device != 'auto':
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def calculate_model_size_mb(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def calculate_flops(self, model, input_size=(1, 3, 256, 256)):
        """Calculate FLOPs using thop"""
        if not THOP_AVAILABLE:
            return None, "thop not available"
        
        try:
            # Create model copy on CPU for FLOPs calculation
            model_cpu = type(model)(n_class=1)
            model_cpu.load_state_dict(model.state_dict())
            model_cpu.eval()
            
            input_cpu = torch.randn(input_size)
            flops, params = profile(model_cpu, inputs=(input_cpu,), verbose=False)
            flops_readable = clever_format([flops], "%.3f")
            return flops, flops_readable[0]
        except Exception as e:
            return None, f"Error: {str(e)[:50]}..."

    def measure_inference_time(self, model, input_tensor, num_runs=50):
        """Measure inference time with warmup"""
        model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / num_runs
        return avg_time, output

    def measure_memory_usage(self, model, input_tensor):
        """Measure GPU memory usage"""
        if self.device.type != 'cuda':
            return None
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        model.eval()
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.memory_allocated() / 1024**2
        memory_used = peak_memory - initial_memory
        
        return memory_used

    def load_model_from_checkpoint(self, checkpoint_path, model_class, file_format='ckpt'):
        """Load model from checkpoint with error handling for different formats"""
        try:
            if not model_class:
                return None, False
                
            model = model_class(n_class=1)
            
            if file_format == 'pth':
                # Handle .pth files (simple state dict)
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            else:
                # Handle .ckpt files (PyTorch Lightning format)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # Handle different state dict formats
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove 'model.' prefix if present
                    new_key = key[6:] if key.startswith('model.') else key
                    new_state_dict[new_key] = value
                
                model.load_state_dict(new_state_dict, strict=False)
            
            model = model.to(self.device)
            return model, True
        except Exception as e:
            print(f"   ❌ Failed to load {checkpoint_path}: {str(e)[:100]}...")
            return None, False

    def extract_checkpoint_metrics(self, checkpoint_path):
        """Extract training metrics from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            metrics = {}
            
            # Extract various metrics from different callback types
            if 'callbacks' in checkpoint:
                for callback_key, callback_data in checkpoint['callbacks'].items():
                    if 'ModelCheckpoint' in callback_key:
                        if 'best_model_score' in callback_data:
                            metrics['best_score'] = float(callback_data['best_model_score'])
                        if 'monitor' in callback_data:
                            metrics['monitor_metric'] = callback_data['monitor']
            
            # Extract training info
            if 'epoch' in checkpoint:
                metrics['epochs_trained'] = checkpoint['epoch'] + 1
            if 'global_step' in checkpoint:
                metrics['training_steps'] = checkpoint['global_step']
            
            # Try to extract learning rate
            if 'lr_schedulers' in checkpoint:
                try:
                    lr_data = checkpoint['lr_schedulers'][0]
                    if 'last_epoch' in lr_data:
                        metrics['last_lr_epoch'] = lr_data['last_epoch']
                except:
                    pass
            
            return metrics
        except Exception as e:
            return {}

class ModelDiscovery:
    def __init__(self, base_path='d:/Research/DermoMamba'):
        self.base_path = Path(base_path)
        
    def discover_all_models(self):
        """Discover all available models from checkpoints and experiments"""
        models = []
        
        # Checkpoints directory - handle different model types
        checkpoints_dir = self.base_path / 'checkpoints'
        if checkpoints_dir.exists():
            for checkpoint_dir in checkpoints_dir.iterdir():
                if checkpoint_dir.is_dir():
                    # Handle .ckpt files
                    checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
                    
                    # Handle .pth files (for tiny models)
                    pth_files = list(checkpoint_dir.glob('*.pth'))
                    
                    # Process .ckpt files
                    for checkpoint_file in checkpoint_files:
                        # Determine model class based on directory name
                        model_class = None
                        if 'tiny' in checkpoint_dir.name.lower():
                            model_class = TinyDermoMamba if TINY_AVAILABLE else None
                        else:
                            model_class = OptimizedDermoMamba if OPTIMIZED_AVAILABLE else None
                            
                        models.append({
                            'name': f"Checkpoint: {checkpoint_dir.name} ({checkpoint_file.name})",
                            'path': str(checkpoint_file),
                            'type': 'checkpoint',
                            'category': checkpoint_dir.name,
                            'model_class': model_class,
                            'file_format': 'ckpt'
                        })
                    
                    # Process .pth files (tiny models)
                    for pth_file in pth_files:
                        models.append({
                            'name': f"Checkpoint: {checkpoint_dir.name} ({pth_file.name})",
                            'path': str(pth_file),
                            'type': 'checkpoint',
                            'category': checkpoint_dir.name,
                            'model_class': TinyDermoMamba if TINY_AVAILABLE else None,
                            'file_format': 'pth'
                        })
        
        # Experiments directory - find ALL best model files
        experiments_dir = self.base_path / 'experiments'
        if experiments_dir.exists():
            for exp_dir in experiments_dir.iterdir():
                if exp_dir.is_dir():
                    # Look for best model files in root of experiment
                    best_files_root = list(exp_dir.glob('best*.ckpt'))
                    
                    # Look for best model files in checkpoints subdirectory
                    checkpoints_subdir = exp_dir / 'checkpoints'
                    best_files_sub = []
                    if checkpoints_subdir.exists():
                        best_files_sub = list(checkpoints_subdir.glob('best*.ckpt'))
                    
                    all_best_files = best_files_root + best_files_sub
                    
                    if all_best_files:
                        # Get the best performing file (highest score in filename if available)
                        def extract_score(filename):
                            import re
                            match = re.search(r'[\d.]+', filename.stem)
                            return float(match.group()) if match else 0
                        
                        best_file = max(all_best_files, key=lambda x: extract_score(x))
                        models.append({
                            'name': f"Experiment: {exp_dir.name[:30]}... ({best_file.name})",
                            'path': str(best_file),
                            'type': 'experiment',
                            'category': exp_dir.name,
                            'model_class': OptimizedDermoMamba if OPTIMIZED_AVAILABLE else None,
                            'file_format': 'ckpt'
                        })
                    else:
                        # If no best files, try to find any model files in checkpoints
                        if checkpoints_subdir.exists():
                            model_files = list(checkpoints_subdir.glob('*.ckpt'))
                            if model_files:
                                # Use final_model.ckpt if available, otherwise the last one
                                final_files = [f for f in model_files if 'final' in f.name]
                                if final_files:
                                    model_file = final_files[0]
                                else:
                                    model_file = max(model_files, key=lambda x: x.stat().st_mtime)
                                
                                models.append({
                                    'name': f"Experiment: {exp_dir.name[:30]}... ({model_file.name})",
                                    'path': str(model_file),
                                    'type': 'experiment',
                                    'category': exp_dir.name,
                                    'model_class': OptimizedDermoMamba if OPTIMIZED_AVAILABLE else None,
                                    'file_format': 'ckpt'
                                })
        
        return models

def comprehensive_comparison():
    print("🔬 COMPREHENSIVE DERMOMAMBA MODEL COMPARISON")
    print("=" * 100)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    profiler = ModelProfiler()
    discovery = ModelDiscovery()
    
    # Discover all models
    print("\n🔍 Discovering models...")
    models = discovery.discover_all_models()
    
    if not models:
        print("❌ No models found!")
        return
    
    print(f"✅ Found {len(models)} models to analyze\n")
    
    # Test configurations
    test_sizes = [
        (1, 3, 224, 224),   # Standard training size
        (1, 3, 256, 256),   # Common inference size
        (1, 3, 384, 384),   # High resolution
        (4, 3, 256, 256),   # Batch inference
    ]
    
    results = []
    
    # Analyze each model
    for i, model_info in enumerate(models, 1):
        print(f"🔍 [{i}/{len(models)}] Analyzing: {model_info['name']}")
        print(f"   📁 Path: {Path(model_info['path']).name}")
        
        if not model_info['model_class']:
            print("   ⚠️  Model class not available, skipping...")
            continue
        
        # Load model
        model, loaded = profiler.load_model_from_checkpoint(
            model_info['path'], 
            model_info['model_class'],
            model_info.get('file_format', 'ckpt')
        )
        
        if not loaded:
            continue
        
        try:
            # Basic metrics
            total_params, trainable_params = profiler.count_parameters(model)
            model_size_mb = profiler.calculate_model_size_mb(model)
            flops_count, flops_readable = profiler.calculate_flops(model)
            checkpoint_metrics = profiler.extract_checkpoint_metrics(model_info['path'])
            
            result = {
                'name': model_info['name'],
                'category': model_info['category'],
                'type': model_info['type'],
                'path': model_info['path'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size_mb,
                'flops_count': flops_count,
                'flops_readable': flops_readable,
                'checkpoint_metrics': checkpoint_metrics,
                'inference_times': {},
                'memory_usage': {},
                'throughput': {},
                'model_loaded': True
            }
            
            # Test different input sizes
            print("   📊 Performance testing...")
            for size in test_sizes:
                size_name = f"{size[0]}x{size[2]}x{size[3]}"
                
                try:
                    input_tensor = torch.randn(size)
                    
                    # Measure inference time
                    avg_time, output = profiler.measure_inference_time(model, input_tensor)
                    result['inference_times'][size_name] = avg_time
                    
                    # Calculate throughput
                    batch_size = size[0]
                    throughput = batch_size / avg_time
                    result['throughput'][size_name] = throughput
                    
                    # Measure memory usage
                    memory_used = profiler.measure_memory_usage(model, input_tensor)
                    if memory_used is not None:
                        result['memory_usage'][size_name] = memory_used
                    
                    print(f"      📏 {size_name}: {avg_time*1000:.1f}ms, {throughput:.1f} img/s")
                    
                except Exception as e:
                    print(f"      ❌ Error with {size_name}: {str(e)[:50]}...")
                    result['inference_times'][size_name] = None
            
            results.append(result)
            
            # Clean up
            del model
            if profiler.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print("   ✅ Analysis complete\n")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {str(e)[:100]}...\n")
    
    # Generate comprehensive report
    generate_comparison_report(results)
    
    return results

def generate_comparison_report(results):
    """Generate comprehensive comparison report"""
    
    if not results:
        print("❌ No results to compare!")
        return
    
    print("\n" + "=" * 120)
    print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 120)
    
    # 1. Model Overview Table
    print(f"\n{'=' * 25} MODEL OVERVIEW {'=' * 25}")
    print(f"{'Model Name':<35} {'Type':<12} {'Params(M)':<12} {'Size(MB)':<12} {'FLOPs':<15}")
    print("-" * 95)
    
    for result in results:
        name = result['name'][:33] + ".." if len(result['name']) > 35 else result['name']
        params_m = result['total_params'] / 1e6
        size_mb = result['model_size_mb']
        flops = result['flops_readable'] if result['flops_readable'] else "N/A"
        
        print(f"{name:<35} {result['type']:<12} {params_m:<12.2f} {size_mb:<12.2f} {flops:<15}")
    
    # 2. Performance Metrics Table
    print(f"\n{'=' * 25} TRAINING PERFORMANCE {'=' * 25}")
    print(f"{'Model Name':<35} {'Best Score':<12} {'Monitor':<12} {'Epochs':<10} {'Steps':<10}")
    print("-" * 85)
    
    for result in results:
        name = result['name'][:33] + ".." if len(result['name']) > 35 else result['name']
        metrics = result['checkpoint_metrics']
        
        best_score = metrics.get('best_score', 'N/A')
        monitor = metrics.get('monitor_metric', 'N/A')
        epochs = metrics.get('epochs_trained', 'N/A')
        steps = metrics.get('training_steps', 'N/A')
        
        if isinstance(best_score, float):
            best_score = f"{best_score:.4f}"
        if isinstance(monitor, str) and len(monitor) > 10:
            monitor = monitor[:8] + ".."
        
        print(f"{name:<35} {best_score:<12} {monitor:<12} {epochs:<10} {steps:<10}")
    
    # 3. Inference Speed Comparison
    print(f"\n{'=' * 25} INFERENCE SPEED (milliseconds) {'=' * 25}")
    print(f"{'Model Name':<35} {'1x224x224':<12} {'1x256x256':<12} {'1x384x384':<12} {'4x256x256':<12}")
    print("-" * 95)
    
    for result in results:
        name = result['name'][:33] + ".." if len(result['name']) > 35 else result['name']
        times = result['inference_times']
        
        row_data = [name]
        for size in ['1x224x224', '1x256x256', '1x384x384', '4x256x256']:
            time_val = times.get(size)
            if time_val is not None:
                row_data.append(f"{time_val*1000:.1f}ms")
            else:
                row_data.append("Failed")
        
        print(f"{row_data[0]:<35} {row_data[1]:<12} {row_data[2]:<12} {row_data[3]:<12} {row_data[4]:<12}")
    
    # 4. Throughput Comparison
    print(f"\n{'=' * 25} THROUGHPUT (images/second) {'=' * 25}")
    print(f"{'Model Name':<35} {'Single':<10} {'Batch':<10} {'High-Res':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    for result in results:
        name = result['name'][:33] + ".." if len(result['name']) > 35 else result['name']
        throughput = result['throughput']
        
        single = throughput.get('1x256x256', 0)
        batch = throughput.get('4x256x256', 0)
        high_res = throughput.get('1x384x384', 0)
        
        # Efficiency metric: throughput per parameter
        efficiency = single / (result['total_params'] / 1e6) if single > 0 else 0
        
        print(f"{name:<35} {single:<10.1f} {batch:<10.1f} {high_res:<10.1f} {efficiency:<12.2f}")
    
    # 5. Memory Usage (if available)
    cuda_results = [r for r in results if r['memory_usage']]
    if cuda_results:
        print(f"\n{'=' * 25} GPU MEMORY USAGE (MB) {'=' * 25}")
        print(f"{'Model Name':<35} {'1x256x256':<12} {'4x256x256':<12} {'1x384x384':<12}")
        print("-" * 75)
        
        for result in cuda_results:
            name = result['name'][:33] + ".." if len(result['name']) > 35 else result['name']
            memory = result['memory_usage']
            
            mem_256 = memory.get('1x256x256', 0)
            mem_batch = memory.get('4x256x256', 0)
            mem_384 = memory.get('1x384x384', 0)
            
            print(f"{name:<35} {mem_256:<12.1f} {mem_batch:<12.1f} {mem_384:<12.1f}")
    
    # 6. Top Performers Analysis
    print(f"\n{'=' * 25} TOP PERFORMERS ANALYSIS {'=' * 25}")
    
    # Best trained models (with checkpoint metrics)
    trained_models = [r for r in results if r['checkpoint_metrics'].get('best_score')]
    if trained_models:
        best_performer = max(trained_models, key=lambda x: x['checkpoint_metrics']['best_score'])
        print(f"🏆 Best Performance: {best_performer['name']}")
        print(f"   Score: {best_performer['checkpoint_metrics']['best_score']:.4f}")
        print(f"   Parameters: {best_performer['total_params']/1e6:.2f}M")
        
        if best_performer['inference_times'].get('1x256x256'):
            speed = best_performer['inference_times']['1x256x256'] * 1000
            throughput = best_performer['throughput'].get('1x256x256', 0)
            print(f"   Speed: {speed:.1f}ms ({throughput:.1f} img/s)")
    
    # Fastest model
    speed_models = [r for r in results if r['inference_times'].get('1x256x256')]
    if speed_models:
        fastest_model = min(speed_models, key=lambda x: x['inference_times']['1x256x256'])
        speed = fastest_model['inference_times']['1x256x256'] * 1000
        throughput = fastest_model['throughput'].get('1x256x256', 0)
        print(f"\n⚡ Fastest Model: {fastest_model['name']}")
        print(f"   Speed: {speed:.1f}ms ({throughput:.1f} img/s)")
        print(f"   Parameters: {fastest_model['total_params']/1e6:.2f}M")
    
    # Most efficient model (performance per parameter)
    if trained_models and speed_models:
        efficiency_models = [r for r in results if r['checkpoint_metrics'].get('best_score') and r['inference_times'].get('1x256x256')]
        if efficiency_models:
            most_efficient = max(efficiency_models, key=lambda x: x['checkpoint_metrics']['best_score'] / (x['total_params']/1e6))
            efficiency_score = most_efficient['checkpoint_metrics']['best_score'] / (most_efficient['total_params']/1e6)
            print(f"\n⚖️ Most Efficient: {most_efficient['name']}")
            print(f"   Efficiency: {efficiency_score:.4f} (score/M-param)")
            print(f"   Score: {most_efficient['checkpoint_metrics']['best_score']:.4f}")
            print(f"   Parameters: {most_efficient['total_params']/1e6:.2f}M")
    
    # 7. Recommendations
    print(f"\n{'=' * 25} DEPLOYMENT RECOMMENDATIONS {'=' * 25}")
    
    if trained_models:
        best_model = max(trained_models, key=lambda x: x['checkpoint_metrics']['best_score'])
        print(f"🏥 Clinical/Research Use: {best_model['name']}")
        print(f"   → Highest accuracy for critical applications")
    
    if speed_models:
        fastest = min(speed_models, key=lambda x: x['inference_times']['1x256x256'])
        print(f"🚀 Real-time Applications: {fastest['name']}")
        print(f"   → Best for interactive systems and real-time processing")
    
    # Most balanced model
    if efficiency_models:
        balanced = max(efficiency_models, key=lambda x: (x['checkpoint_metrics']['best_score'] * 0.7) + (1/x['inference_times']['1x256x256'] * 0.3))
        print(f"⚖️ Balanced Deployment: {balanced['name']}")
        print(f"   → Good balance of accuracy and speed")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_results_{timestamp}.json"
    
    # Prepare serializable data
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Convert any numpy types to native Python types
        for key, value in serializable_result.items():
            if isinstance(value, np.integer):
                serializable_result[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_result[key] = float(value)
        serializable_results.append(serializable_result)
    
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_models': len(results),
                'analysis_config': {
                    'test_sizes': ['1x224x224', '1x256x256', '1x384x384', '4x256x256'],
                    'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                    'thop_available': THOP_AVAILABLE
                },
                'results': serializable_results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting comprehensive model comparison...")
        results = comprehensive_comparison()
        print(f"\n✅ Analysis complete! Compared {len(results)} models.")
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
