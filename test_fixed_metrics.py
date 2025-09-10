import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def dice_score(pred, target, smooth=1e-6):
    """Compute Dice score with hard thresholding for metrics"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    
    # Compute per-sample Dice and then average
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred_i = pred[i].flatten()
        target_i = target[i].flatten()
        
        intersection = (pred_i * target_i).sum()
        dice_i = (2.0 * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
        dice_scores.append(dice_i)
    
    return torch.stack(dice_scores).mean()

# Test the metrics
torch.manual_seed(42)
pred_logits = torch.randn(4, 1, 64, 64) * 2  # Batch of 4, simulate model outputs
target = torch.randint(0, 2, (4, 1, 64, 64)).float()  # Binary masks

dice = dice_score(pred_logits, target)
print(f"Dice score: {dice:.6f}")
print(f"Dice should be between 0 and 1: {0 <= dice <= 1}")

# Test with extreme case
pred_all_zeros = torch.zeros(4, 1, 64, 64) - 10  # Will be sigmoid -> 0
pred_all_ones = torch.zeros(4, 1, 64, 64) + 10   # Will be sigmoid -> 1

dice_zeros = dice_score(pred_all_zeros, target)
dice_ones = dice_score(pred_all_ones, target)

print(f"Dice with all-zero predictions: {dice_zeros:.6f}")
print(f"Dice with all-one predictions: {dice_ones:.6f}")
