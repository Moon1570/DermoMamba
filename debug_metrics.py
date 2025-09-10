#!/usr/bin/env python3
"""
Quick debug script to check the metrics calculation issue
"""
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def dice_score_debug(pred, target, smooth=1e-6):
    """Debug version of dice score calculation"""
    print(f"Input pred shape: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Input target shape: {target.shape}, range: [{target.min():.4f}, {target.max():.4f}]")
    
    pred_sigmoid = torch.sigmoid(pred)
    print(f"After sigmoid pred range: [{pred_sigmoid.min():.4f}, {pred_sigmoid.max():.4f}]")
    
    pred_binary = (pred_sigmoid > 0.5).float()
    print(f"After threshold pred range: [{pred_binary.min():.4f}, {pred_binary.max():.4f}]")
    print(f"Binary pred unique values: {torch.unique(pred_binary)}")
    print(f"Target unique values: {torch.unique(target)}")
    
    intersection = (pred_binary * target).sum()
    pred_sum = pred_binary.sum()
    target_sum = target.sum()
    
    print(f"Intersection: {intersection.item()}")
    print(f"Pred sum: {pred_sum.item()}")
    print(f"Target sum: {target_sum.item()}")
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    print(f"Dice calculation: (2 * {intersection.item()} + {smooth}) / ({pred_sum.item()} + {target_sum.item()} + {smooth}) = {dice.item()}")
    
    return dice

# Test with sample data
torch.manual_seed(42)
pred = torch.randn(2, 1, 4, 4) * 2  # Raw logits
target = torch.randint(0, 2, (2, 1, 4, 4)).float()  # Binary mask

print("=== DEBUG DICE SCORE CALCULATION ===")
dice_result = dice_score_debug(pred, target)
print(f"Final Dice: {dice_result.item():.4f}")
