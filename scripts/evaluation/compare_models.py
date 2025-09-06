"""
Comprehensive Model Comparison: Parameters, FLOPs, Time, IoU, and Performance Metrics
"""
import torch
import torch.nn as nn
import time
import numpy as np
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. FLOPs calculation will be skipped.")

import sys
sys.path.append('.')

# Import all model variants
try:
    from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OptimizedDermoMamba = None
    OPTIMIZED_AVAILABLE = False
    print("Warning: OptimizedDermoMamba not available")

try:
    from module.model.tiny_dermomamba import TinyDermoMamba
    TINY_AVAILABLE = True
except ImportError:
    TinyDermoMamba = None
    TINY_AVAILABLE = False
    print("Warning: TinyDermoMamba not available")

try:
    from module.model.tiny_dermomamba_resmamba import TinyDermoMambaResMamba
    RESMAMBA_AVAILABLE = True
except ImportError:
    TinyDermoMambaResMamba = None
    RESMAMBA_AVAILABLE = False
    print("Warning: TinyDermoMambaResMamba not available")

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def measure_inference_time(model, input_tensor, num_runs=50):
    """Measure inference time with warmup"""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
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
    
    avg_time = (time.time() - start_time) / num_runs
    return avg_time, output

def calculate_flops(model, input_size=(1, 3, 256, 256)):
    """Calculate FLOPs using thop"""
    if not THOP_AVAILABLE:
        return None, "thop not available"
    
    # Get device from model
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_size).to(device)
    
    try:
        # Move model to CPU for FLOPs calculation to avoid device issues
        model_cpu = model.cpu()
        input_cpu = torch.randn(input_size)
        flops, params = profile(model_cpu, inputs=(input_cpu,), verbose=False)
        flops_readable = clever_format([flops], "%.3f")
        # Move model back to original device
        model.to(device)
        return flops, flops_readable[0]
    except Exception as e:
        return None, f"Error: {e}"

def calculate_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def calculate_dice_iou(pred, target, threshold=0.5):
    """Calculate Dice and IoU scores"""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Intersection and Union
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    union = pred_sum + target_sum - intersection
    
    # Dice
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    return dice.item(), iou.item()

def load_checkpoint_metrics(checkpoint_path):
    """Load metrics from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metrics = {}
        
        if 'callbacks' in checkpoint:
            for callback_key, callback_data in checkpoint['callbacks'].items():
                if 'ModelCheckpoint' in callback_key:
                    if 'best_model_score' in callback_data:
                        metrics['best_dice'] = float(callback_data['best_model_score'])
        
        if 'epoch' in checkpoint:
            metrics['epochs'] = checkpoint['epoch'] + 1  # 0-indexed to 1-indexed
        if 'global_step' in checkpoint:
            metrics['steps'] = checkpoint['global_step']
            
        return metrics
    except:
        return {}

def comprehensive_model_comparison():
    print("="*90)
    print("🔬 COMPREHENSIVE DERMOMAMBA MODEL COMPARISON")
    print("="*90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configurations
    models_config = []
    
    if TINY_AVAILABLE:
        models_config.append({
            'name': 'Tiny DermoMamba',
            'model_class': TinyDermoMamba,
            'checkpoint': None,
            'description': 'Lightweight version (no ResMamba)'
        })
    
    if RESMAMBA_AVAILABLE:
        models_config.append({
            'name': 'Tiny ResMamba',
            'model_class': TinyDermoMambaResMamba,
            'checkpoint': None,
            'description': 'ResMamba version (slow but accurate)'
        })
    
    models_config.extend([
        {
            'name': 'Optimized Complete',
            'model_class': OptimizedDermoMamba if OPTIMIZED_AVAILABLE else None,
            'checkpoint': 'checkpoints/optimized_complete/best_model-v1.ckpt',
            'description': 'Optimized with all paper properties'
        },
        {
            'name': 'Improved Complete',
            'model_class': OptimizedDermoMamba if OPTIMIZED_AVAILABLE else None,
            'checkpoint': 'checkpoints/optimized_complete_improved/best_model-v1.ckpt',
            'description': 'Best performing model (89.25% dice)'
        }
    ])
    
    # Test configurations
    test_sizes = [
        (1, 3, 192, 192),
        (1, 3, 256, 256),
        (1, 3, 384, 384),
        (4, 3, 256, 256),  # Batch processing
    ]
    
    results = []
    
    for config in models_config:
        print(f"\n🔍 Analyzing: {config['name']}")
        print(f"   📝 {config['description']}")
        
        try:
            # Create model
            model = config['model_class'](n_class=1).to(device)
            model.eval()
            
            # Basic metrics
            total_params, trainable_params = count_parameters(model)
            model_size_mb = calculate_model_size_mb(model)
            
            # Load checkpoint metrics if available
            checkpoint_metrics = {}
            if config['checkpoint']:
                checkpoint_metrics = load_checkpoint_metrics(config['checkpoint'])
            
            # FLOPs calculation
            flops_count, flops_readable = calculate_flops(model, (1, 3, 256, 256))
            
            model_results = {
                'name': config['name'],
                'description': config['description'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size_mb,
                'flops_count': flops_count,
                'flops_readable': flops_readable,
                'checkpoint_metrics': checkpoint_metrics,
                'inference_times': {},
                'memory_usage': {},
                'performance_metrics': {}
            }
            
            # Test different input sizes
            for size in test_sizes:
                size_name = f"{size[0]}x{size[2]}x{size[3]}"
                print(f"   📊 Testing {size_name}...")
                
                try:
                    input_tensor = torch.randn(size)
                    
                    # Measure inference time
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        initial_memory = torch.cuda.memory_allocated() / 1024**2
                    
                    avg_time, output = measure_inference_time(model, input_tensor)
                    
                    if device.type == 'cuda':
                        peak_memory = torch.cuda.memory_allocated() / 1024**2
                        memory_used = peak_memory - initial_memory
                        model_results['memory_usage'][size_name] = memory_used
                    
                    model_results['inference_times'][size_name] = avg_time
                    
                    # Calculate throughput (images per second)
                    batch_size = size[0]
                    throughput = batch_size / avg_time
                    
                    # For synthetic performance test (dummy target)
                    if size == (1, 3, 256, 256):
                        dummy_target = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32)
                        dice, iou = calculate_dice_iou(output.cpu(), dummy_target)
                        model_results['performance_metrics']['synthetic_dice'] = dice
                        model_results['performance_metrics']['synthetic_iou'] = iou
                    
                    print(f"      ⏱️  Time: {avg_time*1000:.2f}ms")
                    print(f"      🚀 Throughput: {throughput:.1f} img/s")
                    if device.type == 'cuda':
                        print(f"      💾 Memory: {memory_used:.1f}MB")
                
                except Exception as e:
                    print(f"      ❌ Error with {size_name}: {e}")
                    model_results['inference_times'][size_name] = None
            
            results.append(model_results)
            
        except Exception as e:
            print(f"   ❌ Failed to analyze {config['name']}: {e}")
    
    # Print comprehensive comparison table
    print("\n" + "="*90)
    print("📊 COMPREHENSIVE COMPARISON TABLE")
    print("="*90)
    
    # Parameters and Size Comparison
    print(f"\n{'Model Name':<20} {'Params (M)':<12} {'Size (MB)':<12} {'FLOPs':<15} {'Best Dice':<12}")
    print("-" * 75)
    
    for result in results:
        params_m = result['total_params'] / 1e6
        size_mb = result['model_size_mb']
        flops = result['flops_readable'] if result['flops_readable'] else "N/A"
        best_dice = result['checkpoint_metrics'].get('best_dice', 'N/A')
        if isinstance(best_dice, float):
            best_dice = f"{best_dice:.4f}"
        
        print(f"{result['name']:<20} {params_m:<12.2f} {size_mb:<12.2f} {flops:<15} {best_dice:<12}")
    
    # Inference Time Comparison
    print(f"\n🚀 INFERENCE TIME COMPARISON (milliseconds)")
    print("-" * 90)
    
    size_headers = ['Model Name'] + [f"{s[0]}x{s[2]}x{s[3]}" for s in test_sizes]
    print(f"{'Model Name':<20} {'1x192x192':<12} {'1x256x256':<12} {'1x384x384':<12} {'4x256x256':<12}")
    print("-" * 75)
    
    for result in results:
        row = [result['name']]
        for size in test_sizes:
            size_name = f"{size[0]}x{size[2]}x{size[3]}"
            time_ms = result['inference_times'].get(size_name)
            if time_ms is not None:
                row.append(f"{time_ms*1000:.2f}ms")
            else:
                row.append("Failed")
        
        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    # Training Performance Comparison
    print(f"\n🎯 TRAINING PERFORMANCE COMPARISON")
    print("-" * 90)
    print(f"{'Model Name':<20} {'Best Dice':<12} {'Best IoU':<12} {'Epochs':<10} {'Status':<15}")
    print("-" * 75)
    
    for result in results:
        name = result['name']
        metrics = result['checkpoint_metrics']
        
        dice = metrics.get('best_dice', 'N/A')
        if isinstance(dice, float):
            dice_str = f"{dice:.4f}"
            iou_estimate = dice / (2 - dice)  # Rough IoU estimate from Dice
            iou_str = f"{iou_estimate:.4f}"
            
            if dice >= 0.91:
                status = "🥇 Excellent"
            elif dice >= 0.80:
                status = "🥈 Very Good"
            elif dice >= 0.60:
                status = "🥉 Good"
            elif dice >= 0.40:
                status = "⚠️ Poor"
            else:
                status = "❌ Failed"
        else:
            dice_str = "N/A"
            iou_str = "N/A"
            status = "❓ Unknown"
        
        epochs = metrics.get('epochs', 'N/A')
        
        print(f"{name:<20} {dice_str:<12} {iou_str:<12} {epochs:<10} {status:<15}")
    
    # Efficiency Analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS")
    print("-" * 90)
    
    # Find best models in each category
    trained_models = [r for r in results if r['checkpoint_metrics']]
    
    if trained_models:
        best_model = max(trained_models, key=lambda x: x['checkpoint_metrics'].get('best_dice', 0))
        print(f"🏆 Best Performance: {best_model['name']}")
        if 'best_dice' in best_model['checkpoint_metrics']:
            dice_score = best_model['checkpoint_metrics']['best_dice']
            iou_estimate = dice_score / (2 - dice_score)
            print(f"   🎯 Dice Score: {dice_score:.4f} ({dice_score*100:.2f}%)")
            print(f"   🎯 IoU Estimate: {iou_estimate:.4f} ({iou_estimate*100:.2f}%)")
    
    fastest_model = min([r for r in results if r['inference_times'].get('1x256x256')], 
                       key=lambda x: x['inference_times'].get('1x256x256', float('inf')), 
                       default=None)
    if fastest_model:
        print(f"\n⚡ Fastest Inference: {fastest_model['name']}")
        fastest_time = fastest_model['inference_times'].get('1x256x256', 0)
        throughput = 1 / fastest_time
        print(f"   ⏱️ Time: {fastest_time*1000:.2f}ms")
        print(f"   🚀 Throughput: {throughput:.1f} img/s")
    
    smallest_model = min(results, key=lambda x: x['total_params'])
    print(f"\n🪶 Smallest Model: {smallest_model['name']}")
    print(f"   📊 Parameters: {smallest_model['total_params']/1e6:.2f}M")
    print(f"   💾 Size: {smallest_model['model_size_mb']:.2f}MB")
    
    # Performance vs Efficiency Trade-offs
    print(f"\n📈 PERFORMANCE vs EFFICIENCY TRADE-OFFS")
    print("-" * 90)
    
    for result in results:
        name = result['name']
        params_m = result['total_params'] / 1e6
        time_256 = result['inference_times'].get('1x256x256', 0)
        dice = result['checkpoint_metrics'].get('best_dice', 0)
        
        if time_256 and dice:
            efficiency_score = dice / (params_m * time_256)  # Dice per (M params * second)
            print(f"{name}: {efficiency_score:.3f} (Dice/M-param/s)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 90)
    if trained_models:
        best_trained = max(trained_models, key=lambda x: x['checkpoint_metrics'].get('best_dice', 0))
        print(f"🏥 For Clinical Use (Accuracy Priority): {best_trained['name']}")
    
    if fastest_model:
        print(f"⚡ For Real-time Applications: {fastest_model['name']}")
    
    print(f"📱 For Mobile/Edge Deployment: {smallest_model['name']}")
    
    # Paper Comparison
    print(f"\n📄 PAPER COMPARISON")
    print("-" * 90)
    paper_dice = 0.91
    if trained_models:
        best_dice_achieved = max(r['checkpoint_metrics'].get('best_dice', 0) for r in trained_models)
        gap = paper_dice - best_dice_achieved
        print(f"Paper Target Dice: {paper_dice:.4f} ({paper_dice*100:.0f}%)")
        print(f"Best Achieved: {best_dice_achieved:.4f} ({best_dice_achieved*100:.2f}%)")
        print(f"Gap: {gap:.4f} ({gap*100:.2f}% points)")
        
        if gap <= 0.02:  # Within 2%
            print("✅ EXCELLENT: Very close to paper performance!")
        elif gap <= 0.05:  # Within 5%
            print("✅ GOOD: Close to paper performance")
        else:
            print("⚠️  Room for improvement to reach paper performance")
    
    return results
    num_runs = 5
    
    print(f"Test config: batch_size={batch_size}, image_size={image_size}, runs={num_runs}")
    print("-" * 60)
    
    # Test Fast DermoMamba
    try:
        from module.model.fast_dermomamba import FastDermoMamba
        
        fast_model = FastDermoMamba(n_class=1).to(device)
        fast_model.eval()
        
        fast_params = sum(p.numel() for p in fast_model.parameters())
        print(f"Fast DermoMamba parameters: {fast_params:,}")
        
        # Warmup
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        for _ in range(3):
            with torch.no_grad():
                _ = fast_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time fast model
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                fast_output = fast_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        fast_time = (time.time() - start_time) / num_runs
        print(f"Fast DermoMamba avg time: {fast_time:.3f}s")
        print(f"Fast DermoMamba output shape: {fast_output.shape}")
        
    except Exception as e:
        print(f"❌ Fast DermoMamba error: {e}")
        fast_time = float('inf')
        fast_params = 0
    
    print("-" * 60)
    
    # Test Optimized DermoMamba
    if OPTIMIZED_AVAILABLE:
        try:
            opt_model = OptimizedDermoMamba(n_class=1).to(device)
            opt_model.eval()
            
            opt_params = sum(p.numel() for p in opt_model.parameters())
            print(f"Optimized DermoMamba parameters: {opt_params:,}")
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = opt_model(x)
            
            # Time optimized model
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    opt_output = opt_model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            opt_time = (time.time() - start_time) / num_runs
            print(f"Optimized DermoMamba avg time: {opt_time:.3f}s")
            print(f"Optimized DermoMamba output shape: {opt_output.shape}")
            
        except Exception as e:
            print(f"❌ Optimized DermoMamba error: {e}")
            opt_time = float('inf')
            opt_params = 0
    else:
        print("❌ OptimizedDermoMamba not available")
        opt_time = float('inf')
        opt_params = 0
    
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if fast_time < float('inf') and opt_time < float('inf'):
        speedup = opt_time / fast_time
        print(f"Fast DermoMamba:      {fast_time:.3f}s ({fast_params:,} params)")
        print(f"Optimized DermoMamba: {opt_time:.3f}s ({opt_params:,} params)")
        print(f"Speed improvement:    {speedup:.2f}x faster")
        
        param_ratio = opt_params / fast_params if fast_params > 0 else 0
        print(f"Parameter ratio:      {param_ratio:.2f}x")
        
        # Estimate training time
        batches_per_epoch = 1146  # From previous training
        fast_epoch_time = fast_time * batches_per_epoch / 60  # minutes
        opt_epoch_time = opt_time * batches_per_epoch / 60    # minutes
        
        print(f"\nEstimated training time per epoch:")
    return results

if __name__ == "__main__":
    results = comprehensive_model_comparison()
