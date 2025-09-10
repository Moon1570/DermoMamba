import torch
import numpy as np

def soft_dice(pred_logits, target, smooth=1e-6):
    """Compute Dice score with sigmoid (soft)"""
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def hard_dice(pred_logits, target, smooth=1e-6):
    """Compute Dice score with thresholding (hard)"""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

# Test with sample data
torch.manual_seed(42)
pred_logits = torch.randn(1, 1, 100, 100) * 5  # Raw logits like our model
target = torch.randint(0, 2, (1, 1, 100, 100)).float()  # Binary mask

soft_score = soft_dice(pred_logits, target)
hard_score = hard_dice(pred_logits, target)

print(f"Soft Dice: {soft_score:.6f}")
print(f"Hard Dice: {hard_score:.6f}")
print(f"Ratio (hard/soft): {hard_score/soft_score:.2f}x")

# Check sigmoid range
sigmoid_pred = torch.sigmoid(pred_logits)
print(f"\nSigmoid range: [{sigmoid_pred.min():.6f}, {sigmoid_pred.max():.6f}]")
print(f"Raw logits range: [{pred_logits.min():.6f}, {pred_logits.max():.6f}]")
