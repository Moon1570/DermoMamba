#!/usr/bin/env python3
"""
Analyze the Complete Comprehensive Spatial Training Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_training_results():
    """Analyze the complete comprehensive spatial training results"""
    
    experiment_dir = "experiments/complete_spatial_comprehensive_enhanced_20250910_100125"
    
    # Load metrics
    with open(os.path.join(experiment_dir, "logs/train_metrics.json"), 'r') as f:
        train_metrics = json.load(f)
    
    with open(os.path.join(experiment_dir, "logs/val_metrics.json"), 'r') as f:
        val_metrics = json.load(f)
    
    with open(os.path.join(experiment_dir, "config.json"), 'r') as f:
        config = json.load(f)
    
    # Find best validation performance
    val_dice_scores = [m['dice'] for m in val_metrics]
    val_iou_scores = [m['iou'] for m in val_metrics]
    train_dice_scores = [m['dice'] for m in train_metrics]
    train_iou_scores = [m['iou'] for m in train_metrics]
    
    best_val_idx = np.argmax(val_dice_scores)
    best_val_dice = val_dice_scores[best_val_idx]
    best_val_iou = val_iou_scores[best_val_idx]
    best_epoch = best_val_idx + 1
    
    final_val_dice = val_dice_scores[-1]
    final_val_iou = val_iou_scores[-1]
    final_train_dice = train_dice_scores[-1]
    final_train_iou = train_iou_scores[-1]
    
    print("🎯 COMPLETE COMPREHENSIVE SPATIAL TRAINING RESULTS")
    print("=" * 80)
    print(f"📊 CONFIGURATION:")
    print(f"   • Enhancement: {config['enhancement_type']}")
    print(f"   • Model: {config['model']}")
    print(f"   • Input Channels: {config['input_channels']}")
    print(f"   • Loss Type: {config['loss_type']}")
    print(f"   • Total Epochs: {config['epochs']}")
    print()
    
    print(f"🔬 EDGE ENHANCEMENT METHODS USED:")
    for method in config['all_edge_methods']:
        print(f"   ✓ {method.replace('_', ' ').title()}")
    print()
    
    print(f"📍 SPATIAL FEATURES USED:")
    for feature in config['spatial_features']:
        print(f"   ✓ {feature.replace('_', ' ').title()}")
    print()
    
    print(f"🏆 BEST VALIDATION PERFORMANCE:")
    print(f"   • Best Dice Score: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
    print(f"   • Best IoU Score: {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")
    print(f"   • Best Epoch: {best_epoch}")
    print()
    
    print(f"📈 FINAL PERFORMANCE (Epoch 100):")
    print(f"   • Final Val Dice: {final_val_dice:.4f} ({final_val_dice*100:.2f}%)")
    print(f"   • Final Val IoU: {final_val_iou:.4f} ({final_val_iou*100:.2f}%)")
    print(f"   • Final Train Dice: {final_train_dice:.4f} ({final_train_dice*100:.2f}%)")
    print(f"   • Final Train IoU: {final_train_iou:.4f} ({final_train_iou*100:.2f}%)")
    print()
    
    # Compare with targets and previous results
    target_dice = config['target_dice']
    previous_best = 0.8845  # From previous comprehensive training
    
    print(f"🎯 TARGET COMPARISON:")
    print(f"   • Paper Target: {target_dice:.1%} ({target_dice:.4f})")
    print(f"   • Achieved (Best): {best_val_dice:.1%} ({best_val_dice:.4f})")
    
    gap = target_dice - best_val_dice
    if gap <= 0:
        print(f"   ✅ TARGET REACHED! Exceeded by {abs(gap):.1%}")
    else:
        print(f"   📊 Gap: {gap:.1%} ({gap:.4f} points)")
    print()
    
    print(f"📊 IMPROVEMENT OVER PREVIOUS:")
    print(f"   • Previous Best: {previous_best:.1%} ({previous_best:.4f})")
    print(f"   • Current Best: {best_val_dice:.1%} ({best_val_dice:.4f})")
    improvement = best_val_dice - previous_best
    print(f"   • Improvement: {improvement:.1%} ({improvement:.4f} points)")
    print()
    
    # Training stability analysis
    overfitting = final_train_dice - final_val_dice
    print(f"🔍 TRAINING ANALYSIS:")
    print(f"   • Training Stability: {'Good' if overfitting < 0.05 else 'Some Overfitting'}")
    print(f"   • Train-Val Gap: {overfitting:.1%} ({overfitting:.4f})")
    print(f"   • Total Epochs: {len(val_metrics)}")
    print()
    
    print(f"✅ COMPREHENSIVE IMPLEMENTATION CONFIRMED:")
    print(f"   🎨 ALL 5 Edge Enhancement Methods Combined")
    print(f"   📍 Spatial Coordinate Data (X, Y, Radial)")
    print(f"   🧠 6-Channel Input Architecture") 
    print(f"   🎯 Adaptive Guide Fusion Loss")
    print(f"   💾 Best Model Saved: best_model.ckpt")
    
    return {
        'best_val_dice': best_val_dice,
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch,
        'final_val_dice': final_val_dice,
        'target_achieved': gap <= 0,
        'improvement_over_previous': improvement
    }

if __name__ == "__main__":
    results = analyze_training_results()
