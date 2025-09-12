"""
IoU Evaluation Script for DermoMamba Models
Comprehensive analysis of Intersection over Union performance
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import json
from datetime import datetime

# Add project root
sys.path.append('d:/Research/DermoMamba')

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from metric.metrics import dice_score, iou_score

class EvaluationDataset(Dataset):
    def __init__(self, data_root):
        image_dir = os.path.join(data_root, 'train_images')
        mask_dir = os.path.join(data_root, 'train_masks')
        
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.mask_dir = mask_dir
        print(f"📊 Loaded {len(self.image_files)} samples for evaluation")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{img_name}_segmentation.png")
        
        # Load and preprocess
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        
        # Resize to model input size
        from PIL import Image as PILImage
        image = PILImage.fromarray(image).resize((224, 224))
        mask = PILImage.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))
        
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        
        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask, img_name

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    model = OptimizedDermoMamba(n_class=1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remove model. prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[6:] if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

def evaluate_model_iou(model, dataloader, device='cuda'):
    """Comprehensive IoU evaluation"""
    model.eval()
    model = model.to(device)
    
    dice_scores = []
    iou_scores = []
    sample_names = []
    
    print("🔍 Starting IoU evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, masks, names) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Calculate metrics for each sample in batch
            for i in range(outputs.shape[0]):
                output_sample = outputs[i:i+1]
                mask_sample = masks[i:i+1]
                
                dice = dice_score(output_sample, mask_sample)
                iou = iou_score(output_sample, mask_sample)
                
                dice_scores.append(float(dice))
                iou_scores.append(float(iou))
                sample_names.append(names[i])
            
            if batch_idx % 10 == 0:
                print(f"📊 Processed {batch_idx * len(images)}/{len(dataloader.dataset)} samples")
    
    return {
        'dice_scores': dice_scores,
        'iou_scores': iou_scores,
        'sample_names': sample_names
    }

def analyze_results(results):
    """Comprehensive analysis of IoU results"""
    dice_scores = results['dice_scores']
    iou_scores = results['iou_scores']
    
    # Calculate statistics
    dice_stats = {
        'mean': np.mean(dice_scores),
        'std': np.std(dice_scores),
        'min': np.min(dice_scores),
        'max': np.max(dice_scores),
        'median': np.median(dice_scores)
    }
    
    iou_stats = {
        'mean': np.mean(iou_scores),
        'std': np.std(iou_scores),
        'min': np.min(iou_scores),
        'max': np.max(iou_scores),
        'median': np.median(iou_scores)
    }
    
    # Performance thresholds
    dice_excellent = sum(1 for d in dice_scores if d > 0.9)
    dice_good = sum(1 for d in dice_scores if 0.8 < d <= 0.9)
    dice_fair = sum(1 for d in dice_scores if 0.7 < d <= 0.8)
    
    iou_excellent = sum(1 for i in iou_scores if i > 0.85)
    iou_good = sum(1 for i in iou_scores if 0.7 < i <= 0.85)
    iou_fair = sum(1 for i in iou_scores if 0.6 < i <= 0.7)
    
    total_samples = len(dice_scores)
    
    return {
        'dice_stats': dice_stats,
        'iou_stats': iou_stats,
        'performance_breakdown': {
            'dice': {
                'excellent (>90%)': f"{dice_excellent}/{total_samples} ({dice_excellent/total_samples*100:.1f}%)",
                'good (80-90%)': f"{dice_good}/{total_samples} ({dice_good/total_samples*100:.1f}%)",
                'fair (70-80%)': f"{dice_fair}/{total_samples} ({dice_fair/total_samples*100:.1f}%)"
            },
            'iou': {
                'excellent (>85%)': f"{iou_excellent}/{total_samples} ({iou_excellent/total_samples*100:.1f}%)",
                'good (70-85%)': f"{iou_good}/{total_samples} ({iou_good/total_samples*100:.1f}%)",
                'fair (60-70%)': f"{iou_fair}/{total_samples} ({iou_fair/total_samples*100:.1f}%)"
            }
        },
        'correlation': np.corrcoef(dice_scores, iou_scores)[0, 1]
    }

def main():
    print("🎯 IoU PERFORMANCE EVALUATION")
    print("📊 Analyzing current model IoU performance")
    print("="*50)
    
    # Configuration
    data_path = 'd:/Research/DermoMamba/data/ISIC2018_test'
    
    # Model paths to evaluate
    models_to_evaluate = [
        {
            'name': 'Optimized Complete Baseline',
            'path': 'd:/Research/DermoMamba/checkpoints/optimized_complete_improved/best_model-v1.ckpt'
        },
        {
            'name': 'Recent Fine-tuned (94.50% Dice)',
            'path': 'd:/Research/DermoMamba/experiments/finetune_20250911_014557/best-epoch=07-val_dice=0.9450.ckpt'
        }
    ]
    
    # Dataset
    dataset = EvaluationDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    all_results = {}
    
    for model_info in models_to_evaluate:
        model_name = model_info['name']
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
            
        print(f"\n🔍 Evaluating: {model_name}")
        print(f"📁 Path: {model_path}")
        
        try:
            # Load and evaluate model
            model = load_model(model_path)
            results = evaluate_model_iou(model, dataloader, device)
            analysis = analyze_results(results)
            
            all_results[model_name] = {
                'path': model_path,
                'results': results,
                'analysis': analysis
            }
            
            # Display results
            dice_stats = analysis['dice_stats']
            iou_stats = analysis['iou_stats']
            
            print(f"\n📊 RESULTS for {model_name}:")
            print(f"🎯 Dice Score: {dice_stats['mean']:.4f} ± {dice_stats['std']:.4f}")
            print(f"🎯 IoU Score:  {iou_stats['mean']:.4f} ± {iou_stats['std']:.4f}")
            print(f"📈 Dice Range: {dice_stats['min']:.4f} - {dice_stats['max']:.4f}")
            print(f"📈 IoU Range:  {iou_stats['min']:.4f} - {iou_stats['max']:.4f}")
            print(f"🔗 Dice-IoU Correlation: {analysis['correlation']:.4f}")
            
            print(f"\n📈 Performance Breakdown:")
            print(f"   Dice - {analysis['performance_breakdown']['dice']['excellent (>90%)']}")
            print(f"   IoU  - {analysis['performance_breakdown']['iou']['excellent (>85%)']}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"iou_evaluation_results_{timestamp}.json"
    
    # Prepare serializable data
    serializable_results = {}
    for model_name, model_data in all_results.items():
        serializable_results[model_name] = {
            'path': model_data['path'],
            'analysis': model_data['analysis']
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n🏆 MODEL COMPARISON SUMMARY:")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Dice':<10} {'IoU':<10}")
        print(f"{'='*60}")
        
        for model_name, data in all_results.items():
            dice_mean = data['analysis']['dice_stats']['mean']
            iou_mean = data['analysis']['iou_stats']['mean']
            print(f"{model_name[:28]:<30} {dice_mean:.4f}     {iou_mean:.4f}")
    
    print(f"\n🎯 NEXT STEPS:")
    if all_results:
        best_iou = max(data['analysis']['iou_stats']['mean'] for data in all_results.values())
        if best_iou < 0.92:
            print(f"🚀 Current best IoU: {best_iou:.4f} (Target: 0.92+)")
            print(f"💡 Recommendation: Run dual-metric fine-tuning for IoU optimization!")
        else:
            print(f"✅ Excellent IoU performance achieved: {best_iou:.4f}")

if __name__ == "__main__":
    main()
