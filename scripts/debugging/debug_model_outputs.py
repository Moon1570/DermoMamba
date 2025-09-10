import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from datasets.isic_dataset import ISICDataset
from torch.utils.data import DataLoader

def debug_model_outputs():
    print("🔍 DEBUGGING MODEL OUTPUTS")
    print("="*50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedDermoMamba(n_class=1)
    model = model.to(device)
    
    # Load some data
    try:
        # Load best checkpoint if available
        checkpoint_path = "experiments/guide_fusion_boundary_enhanced_20250909_000307/checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Create dataset
        val_dataset = ISICDataset(
            data_root="data/ISIC2018_proc",
            split_file="splits/isic2018_val.txt",
            is_train=False
        )
        
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        model.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                if i >= 2:  # Only check first 2 batches
                    break
                    
                images = images.to(device)
                masks = masks.to(device)
                
                print(f"\n📊 Batch {i+1}:")
                print(f"  Input shape: {images.shape}")
                print(f"  Input range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"  Mask shape: {masks.shape}")
                print(f"  Mask range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
                print(f"  Mask unique values: {torch.unique(masks).cpu().numpy()}")
                
                # Get model output
                outputs = model(images)
                print(f"  Output shape: {outputs.shape}")
                print(f"  Raw output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                
                # Apply sigmoid
                sig_outputs = torch.sigmoid(outputs)
                print(f"  Sigmoid output range: [{sig_outputs.min().item():.4f}, {sig_outputs.max().item():.4f}]")
                
                # Apply threshold
                pred_binary = (sig_outputs > 0.5).float()
                print(f"  Binary pred unique values: {torch.unique(pred_binary).cpu().numpy()}")
                
                # Calculate basic metrics
                intersection = (pred_binary * masks).sum().item()
                union = (pred_binary + masks).clamp(0, 1).sum().item()
                pred_sum = pred_binary.sum().item()
                mask_sum = masks.sum().item()
                
                dice = (2 * intersection) / (pred_sum + mask_sum + 1e-8)
                iou = intersection / (union + 1e-8)
                
                print(f"  Intersection: {intersection}")
                print(f"  Union: {union}")
                print(f"  Pred sum: {pred_sum}")
                print(f"  Mask sum: {mask_sum}")
                print(f"  Dice: {dice:.6f}")
                print(f"  IoU: {iou:.6f}")
                
                # Check if model is predicting anything
                if pred_sum == 0:
                    print("  ⚠️  MODEL IS NOT PREDICTING ANYTHING!")
                elif pred_sum == pred_binary.numel():
                    print("  ⚠️  MODEL IS PREDICTING EVERYTHING!")
                else:
                    print(f"  ✓ Model predicting {pred_sum/pred_binary.numel()*100:.2f}% positive")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_outputs()
