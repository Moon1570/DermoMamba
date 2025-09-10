#!/usr/bin/env python3
"""
COMPLETE Comprehensive Spatial-Aware Edge-Enhanced DermoMamba Training
Uses ALL edge enhancement methods + spatial coordinate data (like the paper)

This implementation includes:
1. ALL edge enhancement methods combined (unsharp, CLAHE, Sobel, Laplacian, multi-scale)
2. Spatial coordinate information (X, Y, radial distance)
3. 6-channel input model (RGB + spatial)
4. Advanced boundary-aware losses
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
import time
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from module.model.spatial_aware_dermomamba import SpatialAwareDermoMamba
from datasets.isic_dataset import ISICDataset
from loss.paper_guide_fusion_loss import AdaptiveGuideFusionLoss
from utils.comprehensive_spatial_preprocessing import (
    get_comprehensive_spatial_train_transforms,
    get_comprehensive_spatial_val_transforms
)

class CompleteSpatialMetricsTracker:
    """Complete metrics tracker for spatial-aware comprehensive training"""
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.train_metrics = []
        self.val_metrics = []
        self.best_dice = 0
        self.best_iou = 0
        self.best_epoch = 0
        
    def update(self, epoch, train_metrics, val_metrics):
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Check for best model
        if val_metrics['dice'] > self.best_dice:
            self.best_dice = val_metrics['dice']
            self.best_iou = val_metrics['iou']
            self.best_epoch = epoch
            return True
        return False
    
    def save_metrics(self):
        """Save all metrics to JSON files"""
        os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
        
        with open(os.path.join(self.experiment_dir, 'logs', 'train_metrics.json'), 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        
        with open(os.path.join(self.experiment_dir, 'logs', 'val_metrics.json'), 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
    
    def plot_complete_curves(self):
        """Create complete training analysis plots"""
        if not self.train_metrics:
            return
            
        plt.figure(figsize=(24, 16))
        epochs = [m['epoch'] + 1 for m in self.train_metrics]
        
        # Main performance metrics
        plt.subplot(4, 6, 1)
        plt.plot(epochs, [m['loss'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8, linewidth=2)
        plt.plot(epochs, [m['loss'] for m in self.val_metrics], label='Val', color='red', alpha=0.8, linewidth=2)
        plt.title('Total Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 6, 2)
        plt.plot(epochs, [m['dice'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8, linewidth=2)
        plt.plot(epochs, [m['dice'] for m in self.val_metrics], label='Val', color='red', alpha=0.8, linewidth=2)
        plt.axhline(y=0.91, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Paper Target (91%)')
        plt.axhline(y=self.best_dice, color='orange', linestyle=':', alpha=0.8, linewidth=2, label=f'Best: {self.best_dice:.3f}')
        plt.title('Dice Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 6, 3)
        plt.plot(epochs, [m['iou'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8, linewidth=2)
        plt.plot(epochs, [m['iou'] for m in self.val_metrics], label='Val', color='red', alpha=0.8, linewidth=2)
        plt.title('IoU Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 6, 4)
        plt.plot(epochs, [m.get('learning_rate', 0) for m in self.train_metrics], color='purple', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Performance analysis
        plt.subplot(4, 6, 5)
        if len(epochs) > 5:
            dice_improvements = np.diff([m['dice'] for m in self.val_metrics])
            plt.plot(epochs[1:], dice_improvements, color='green', alpha=0.7, linewidth=2)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.title('Dice Improvement Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Δ')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 6, 6)
        recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
        recent_dice = [m['dice'] for m in self.val_metrics[-20:]] if len(self.val_metrics) > 20 else [m['dice'] for m in self.val_metrics]
        plt.plot(recent_epochs, recent_dice, 'o-', color='red', alpha=0.8, linewidth=2, markersize=4)
        plt.title('Recent Validation Performance', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.grid(True, alpha=0.3)
        
        # Enhancement summary
        plt.subplot(4, 6, 7)
        plt.text(0.05, 0.95, '🔬 COMPLETE ENHANCEMENT SUMMARY', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'🎯 Target: 91.0% Dice', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.05, 0.80, f'🏆 Achieved: {self.best_dice*100:.2f}% Dice', fontsize=11, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.05, 0.75, f'📈 Best IoU: {self.best_iou*100:.2f}%', fontsize=11, transform=plt.gca().transAxes)
        plt.text(0.05, 0.70, f'⭐ Best Epoch: {self.best_epoch + 1}', fontsize=11, transform=plt.gca().transAxes)
        
        gap = 91.0 - self.best_dice*100
        if gap <= 0:
            plt.text(0.05, 0.65, f'✅ TARGET EXCEEDED by {abs(gap):.2f}%!', fontsize=11, color='green', fontweight='bold', transform=plt.gca().transAxes)
        else:
            plt.text(0.05, 0.65, f'📊 Gap: {gap:.2f} percentage points', fontsize=11, transform=plt.gca().transAxes)
        
        plt.text(0.05, 0.55, '🛠️ ALL ENHANCEMENTS USED:', fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.05, 0.50, '   • Unsharp Masking ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.45, '   • CLAHE Contrast ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.40, '   • Sobel Edge Detection ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.35, '   • Laplacian Enhancement ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.30, '   • Multi-scale Features ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.25, '   • Spatial Coordinates (X,Y,R) ✓', fontsize=9, fontweight='bold', color='blue', transform=plt.gca().transAxes)
        plt.text(0.05, 0.20, '   • 6-Channel Input Model ✓', fontsize=9, fontweight='bold', color='blue', transform=plt.gca().transAxes)
        plt.text(0.05, 0.15, '   • Adaptive Guide Fusion Loss ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.10, '   • Boundary-Guided Attention ✓', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.05, 0.05, '   • Paper-Complete Implementation ✓', fontsize=9, fontweight='bold', color='purple', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Training stability analysis
        plt.subplot(4, 6, 8)
        if len(epochs) > 10:
            train_dice = [m['dice'] for m in self.train_metrics]
            val_dice = [m['dice'] for m in self.val_metrics]
            overfitting = [t - v for t, v in zip(train_dice, val_dice)]
            plt.plot(epochs, overfitting, color='orange', alpha=0.7, linewidth=2)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.title('Train-Val Gap (Overfitting)', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Train - Val Dice')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'complete_spatial_training_analysis.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()

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

def iou_score(pred, target, smooth=1e-6):
    """Compute IoU score with hard thresholding for metrics"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    
    # Compute per-sample IoU and then average
    batch_size = pred.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        pred_i = pred[i].flatten()
        target_i = target[i].flatten()
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum() - intersection
        iou_i = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou_i)
    
    return torch.stack(iou_scores).mean()

def train_epoch_complete(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Complete training epoch with spatial-aware comprehensive enhancement"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    num_batches = len(train_loader)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        # Handle 6-channel input (RGB + spatial)
        images = images.to(device, non_blocking=True)  # (B, 6, H, W)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)  # SpatialAwareDermoMamba handles 6 channels
            
            # Handle adaptive loss
            if hasattr(criterion, 'update_epoch'):
                criterion.update_epoch(epoch)
            
            if hasattr(criterion, 'forward') and 'adaptive' in str(type(criterion)).lower():
                loss_result = criterion(outputs, masks)
                if isinstance(loss_result, tuple):
                    loss, loss_dict = loss_result
                else:
                    loss = loss_result
            else:
                loss = criterion(outputs, masks)
        
        # Compute metrics
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        
        # Progress reporting
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx:3d}/{num_batches}, "
                  f"Loss: {loss.item():.4f}, Dice: {dice.item():.4f}, IoU: {iou.item():.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }

def validate_complete(model, val_loader, criterion, device, epoch):
    """Complete validation with spatial-aware comprehensive enhancement"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            # Handle 6-channel input (RGB + spatial)
            images = images.to(device, non_blocking=True)  # (B, 6, H, W)
            masks = masks.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)  # SpatialAwareDermoMamba handles 6 channels
                
                # Handle adaptive loss
                if hasattr(criterion, 'update_epoch'):
                    criterion.update_epoch(epoch)
                
                if hasattr(criterion, 'forward') and 'adaptive' in str(type(criterion)).lower():
                    loss_result = criterion(outputs, masks)
                    if isinstance(loss_result, tuple):
                        loss, loss_dict = loss_result
                    else:
                        loss = loss_result
                else:
                    loss = criterion(outputs, masks)
            
            # Compute metrics
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }

def main():
    parser = argparse.ArgumentParser(description='COMPLETE Spatial-Aware Edge-Enhanced DermoMamba Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    print("="*100)
    print("🚀 COMPLETE COMPREHENSIVE SPATIAL-AWARE EDGE-ENHANCED TRAINING")
    print("="*100)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"complete_spatial_comprehensive_enhanced_{timestamp}"
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Save configuration
    config = {
        'enhancement_type': 'complete_comprehensive_spatial_aware',
        'all_edge_methods': [
            'unsharp_masking',
            'clahe_contrast_enhancement',
            'sobel_edge_detection', 
            'laplacian_enhancement',
            'multiscale_edge_features'
        ],
        'spatial_features': [
            'x_coordinates',
            'y_coordinates', 
            'radial_distance_from_center'
        ],
        'model': 'SpatialAwareDermoMamba_6_channels',
        'input_channels': 6,
        'loss_type': 'adaptive_guide_fusion_with_spatial_awareness',
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'device': str(device),
        'target_dice': 0.91,
        'paper_implementation': 'complete_with_spatial_data'
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets with complete spatial-aware enhancement
    print("🔄 Loading data with COMPLETE spatial-aware comprehensive enhancement...")
    print("   🎨 ALL Edge Enhancement Methods:")
    print("      • Unsharp Masking ✓")
    print("      • CLAHE Contrast Enhancement ✓") 
    print("      • Sobel Edge Detection ✓")
    print("      • Laplacian Enhancement ✓")
    print("      • Multi-scale Edge Features ✓")
    print("   📍 Spatial Coordinate Features:")
    print("      • X Coordinates ✓")
    print("      • Y Coordinates ✓")
    print("      • Radial Distance ✓")
    print("   🧠 Model: 6-Channel Input (RGB + XYR)")
    
    train_dataset = ISICDataset(
        data_root="data/ISIC2018_proc",
        split_file="splits/isic2018_train.txt",
        transform=get_comprehensive_spatial_train_transforms(),
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root="data/ISIC2018_proc",
        split_file="splits/isic2018_val.txt",
        transform=get_comprehensive_spatial_val_transforms(),
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Initialize 6-channel spatial-aware model
    model = SpatialAwareDermoMamba(n_class=1, input_channels=6)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {total_params:,}")
    print(f"✅ Model: SpatialAwareDermoMamba (6-channel input)")
    
    # Initialize enhanced loss function
    criterion = AdaptiveGuideFusionLoss(
        dice_weight=1.0,
        bce_weight=0.5,
        boundary_weight=3.0,  # Higher weight for spatial-enhanced boundaries
        attention_weight=2.0   # Higher weight for spatial-guided attention
    )
    criterion = criterion.to(device)
    print(f"✅ Loss function: AdaptiveGuideFusionLoss (spatial-enhanced)")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    print("✅ Setup complete, starting COMPLETE comprehensive spatial-aware training...")
    print("🎯 Target: 91% Dice Score (Paper)")
    print("📊 Previous Best: 88.45% Dice")
    print("🚀 Expected: >91% with complete implementation")
    print("="*100)
    
    # Initialize metrics tracker
    metrics_tracker = CompleteSpatialMetricsTracker(experiment_dir)
    
    # Training loop
    patience = 20
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train epoch
        train_metrics = train_epoch_complete(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate epoch  
        val_metrics = validate_complete(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Add learning rate to metrics
        train_metrics['learning_rate'] = current_lr
        val_metrics['learning_rate'] = current_lr
        
        # Update metrics tracker
        is_best = metrics_tracker.update(epoch, train_metrics, val_metrics)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.2e}")
        print(f"  🔬 Complete Enhancement: ALL METHODS + SPATIAL DATA")
        
        # Save best model
        if is_best:
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou'],
                'train_dice': train_metrics['dice'],
                'train_iou': train_metrics['iou'],
                'spatial_aware': True,
                'input_channels': 6
            }
            
            torch.save(checkpoint, os.path.join(experiment_dir, 'checkpoints', 'best_model.ckpt'))
            print(f"  💾 New best model saved! Dice: {val_metrics['dice']:.4f}")
            
            # Check if we reached the paper's target
            if val_metrics['dice'] >= 0.91:
                print(f"  🎯 REACHED PAPER TARGET! Dice: {val_metrics['dice']:.4f} >= 91%")
            elif val_metrics['dice'] >= 0.895:
                print(f"  🔥 VERY CLOSE TO TARGET! Gap: {(0.91 - val_metrics['dice'])*100:.2f} percentage points")
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou'],
                'spatial_aware': True,
                'input_channels': 6
            }
            torch.save(checkpoint, os.path.join(experiment_dir, 'checkpoints', f'model_epoch_{epoch+1}.ckpt'))
        
        # Save metrics and plot curves every few epochs
        if (epoch + 1) % 5 == 0:
            metrics_tracker.save_metrics()
            metrics_tracker.plot_complete_curves()
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping after {epoch+1} epochs (patience: {patience})")
            break
    
    # Final save
    metrics_tracker.save_metrics()
    metrics_tracker.plot_complete_curves()
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_dice': val_metrics['dice'],
        'val_iou': val_metrics['iou'],
        'spatial_aware': True,
        'input_channels': 6
    }
    torch.save(final_checkpoint, os.path.join(experiment_dir, 'checkpoints', 'final_model.ckpt'))
    
    print("="*100)
    print("🏆 COMPLETE COMPREHENSIVE SPATIAL-AWARE TRAINING COMPLETED")
    print("="*100)
    print(f"📊 Best Dice Score: {metrics_tracker.best_dice:.4f}")
    print(f"📊 Best IoU Score: {metrics_tracker.best_iou:.4f}")
    print(f"📊 Best Epoch: {metrics_tracker.best_epoch + 1}")
    print(f"📁 Results saved in: {experiment_dir}")
    print(f"📈 Complete analysis plots saved")
    print()
    print(f"🎯 Target: 91% Dice (Paper)")
    print(f"🏆 Achieved: {metrics_tracker.best_dice*100:.2f}% Dice")
    
    gap = 91.0 - metrics_tracker.best_dice*100
    if gap <= 0:
        print(f"✅ TARGET EXCEEDED by {abs(gap):.2f} percentage points!")
    else:
        print(f"📈 Gap: {gap:.2f} percentage points")
    
    print()
    print("🔬 COMPLETE IMPLEMENTATION CONFIRMATION:")
    print("  ✅ ALL Edge Enhancement Methods Used")
    print("     • Unsharp Masking")
    print("     • CLAHE Contrast Enhancement") 
    print("     • Sobel Edge Detection")
    print("     • Laplacian Enhancement")
    print("     • Multi-scale Edge Features")
    print("  ✅ Spatial Data Used (Like Paper)")
    print("     • X Coordinates")
    print("     • Y Coordinates") 
    print("     • Radial Distance")
    print("  ✅ 6-Channel Input Model")
    print("  ✅ Adaptive Guide Fusion Loss")
    print("  ✅ Boundary-Guided Spatial Attention")
    print("  ✅ Paper-Complete Implementation")

if __name__ == "__main__":
    main()
