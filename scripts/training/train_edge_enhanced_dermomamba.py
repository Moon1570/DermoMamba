#!/usr/bin/env python3
"""
Enhanced DermoMamba Training with Edge-Enhanced Preprocessing and Boundary Loss
This script trains DermoMamba with edge-enhanced image preprocessing specifically
designed to improve boundary detection and boundary loss performance.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from datasets.isic_dataset import create_data_loaders
from loss.paper_guide_fusion_loss import GuideFusionLoss, AdaptiveGuideFusionLoss
from loss.enhanced_boundary_loss import AdvancedBoundaryLoss
from utils.edge_preprocessing import visualize_edge_enhancement

class EdgeEnhancedMetricsTracker:
    """Enhanced metrics tracker with edge-specific monitoring"""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.train_metrics = []
        self.val_metrics = []
        self.best_dice = 0
        self.best_epoch = 0
        self.edge_quality_scores = []
        
    def update(self, train_metrics, val_metrics, epoch):
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Check for best model
        if val_metrics['dice'] > self.best_dice:
            self.best_dice = val_metrics['dice']
            self.best_epoch = epoch
            return True
        return False
    
    def save_metrics(self):
        """Save comprehensive metrics"""
        metrics_data = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_dice': float(self.best_dice),
            'best_epoch': int(self.best_epoch),
            'total_epochs': len(self.train_metrics)
        }
        
        with open(os.path.join(self.experiment_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def plot_enhanced_metrics(self):
        """Create enhanced visualization with edge-specific metrics"""
        epochs = list(range(1, len(self.train_metrics) + 1))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training vs Validation Loss
        axes[0, 0].plot(epochs, [m['loss'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8)
        axes[0, 0].plot(epochs, [m['loss'] for m in self.val_metrics], label='Val', color='red', alpha=0.8)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Score
        axes[0, 1].plot(epochs, [m['dice'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8)
        axes[0, 1].plot(epochs, [m['dice'] for m in self.val_metrics], label='Val', color='red', alpha=0.8)
        axes[0, 1].axhline(y=0.91, color='green', linestyle='--', alpha=0.7, label='Target (91%)')
        axes[0, 1].set_title('Dice Score Progress', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU Score
        axes[0, 2].plot(epochs, [m['iou'] for m in self.train_metrics], label='Train', color='blue', alpha=0.8)
        axes[0, 2].plot(epochs, [m['iou'] for m in self.val_metrics], label='Val', color='red', alpha=0.8)
        axes[0, 2].set_title('IoU Score Progress', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss Components (if available)
        if 'boundary_loss' in self.train_metrics[0]:
            axes[1, 0].plot(epochs, [m.get('dice_loss', 0) for m in self.train_metrics], 
                           label='Dice Loss', alpha=0.8)
            axes[1, 0].plot(epochs, [m.get('bce_loss', 0) for m in self.train_metrics], 
                           label='BCE Loss', alpha=0.8)
            axes[1, 0].plot(epochs, [m.get('boundary_loss', 0) for m in self.train_metrics], 
                           label='Boundary Loss', alpha=0.8)
            axes[1, 0].set_title('Loss Components (Train)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, [m['learning_rate'] for m in self.train_metrics], color='purple', alpha=0.8)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 2].text(0.1, 0.9, f'Best Dice: {self.best_dice:.4f}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.8, f'Best Epoch: {self.best_epoch}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f'Final Dice: {self.val_metrics[-1]["dice"]:.4f}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f'Total Epochs: {len(epochs)}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f'Edge Enhancement: ✓ Enabled', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.3, f'Boundary Loss: ✓ Active', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Training Summary', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'enhanced_training_curves.png'), 
                   dpi=150, bbox_inches='tight')
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

def train_epoch_enhanced(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Enhanced training epoch with edge-aware processing"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # Track edge enhancement effects
    edge_boundary_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            
            # Compute loss (Guide Fusion Loss with boundary awareness)
            if hasattr(criterion, 'forward') and 'GuideFusion' in criterion.__class__.__name__:
                # Guide Fusion Loss returns tuple (loss, loss_dict)
                loss_result = criterion(outputs, masks)
                if isinstance(loss_result, tuple):
                    loss, loss_dict = loss_result
                    edge_boundary_loss += loss_dict.get('boundary_loss', 0)
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
        
        # Progress reporting with edge enhancement info
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}/{num_batches}, "
                  f"Loss: {loss.item():.4f}, Dice: {dice.item():.4f}, IoU: {iou.item():.4f}")
    
    # Prepare metrics
    metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'edge_boundary_loss': edge_boundary_loss / num_batches
    }
    
    return metrics

def validate_enhanced(model, val_loader, criterion, device):
    """Enhanced validation with edge-aware metrics"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                # Handle Guide Fusion Loss tuple return
                loss_result = criterion(outputs, masks)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]  # Extract just the loss tensor
                else:
                    loss = loss_result
            
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
    
    num_batches = len(val_loader)
    metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Enhanced DermoMamba Training with Edge Enhancement')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--edge_method', type=str, default='boundary_optimized', 
                       choices=['boundary_optimized', 'unsharp_mask', 'clahe_edges', 'sobel'],
                       help='Edge enhancement method')
    parser.add_argument('--loss_type', type=str, default='adaptive_guide_fusion',
                       choices=['guide_fusion', 'adaptive_guide_fusion', 'advanced_boundary'],
                       help='Loss function type')
    args = parser.parse_args()

    print("="*80)
    print("🔬 EDGE-ENHANCED DERMOMAMBA TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"edge_enhanced_boundary_{args.edge_method}_{timestamp}"
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Save configuration
    config = {
        'edge_enhancement': args.edge_method,
        'loss_type': args.loss_type,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'device': str(device),
        'model': 'OptimizedDermoMamba',
        'edge_enhanced_preprocessing': True
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders with edge enhancement
    print("🔄 Loading data with edge enhancement...")
    train_loader, val_loader = create_data_loaders(
        data_root="data/ISIC2018_proc",
        train_split="splits/isic2018_train.txt",
        val_split="splits/isic2018_val.txt",
        batch_size=args.batch_size,
        num_workers=4,
        edge_enhanced=True
    )
    
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = OptimizedDermoMamba(n_class=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {total_params:,}")
    
    # Setup loss function
    if args.loss_type == 'guide_fusion':
        criterion = GuideFusionLoss()
    elif args.loss_type == 'adaptive_guide_fusion':
        criterion = AdaptiveGuideFusionLoss(dice_weight=1.0, bce_weight=0.5, boundary_weight=2.0)
    else:  # advanced_boundary
        criterion = AdvancedBoundaryLoss()
    
    criterion = criterion.to(device)
    print(f"✅ Loss function: {args.loss_type}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Setup metrics tracker
    metrics_tracker = EdgeEnhancedMetricsTracker(experiment_dir)
    
    print("✅ Setup complete, starting enhanced training...")
    print("="*80)
    print()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Update adaptive loss weights if using adaptive loss
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch_enhanced(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate
        val_metrics = validate_enhanced(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Update metrics
        is_best = metrics_tracker.update(train_metrics, val_metrics, epoch)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {train_metrics['learning_rate']:.2e}")
        print(f"  Edge Enhancement: {args.edge_method}")
        
        if is_best:
            print(f"  💾 New best model saved! Dice: {val_metrics['dice']:.4f}")
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dice_score': val_metrics['dice'],
                'config': config
            }, os.path.join(experiment_dir, 'checkpoints', 'best_model.ckpt'))
        
        # Save periodic checkpoints
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dice_score': val_metrics['dice'],
                'config': config
            }, os.path.join(experiment_dir, 'checkpoints', f'model_epoch_{epoch}.ckpt'))
        
        print()
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'dice_score': val_metrics['dice'],
        'config': config
    }, os.path.join(experiment_dir, 'checkpoints', 'final_model.ckpt'))
    
    # Save metrics and create plots
    metrics_tracker.save_metrics()
    metrics_tracker.plot_enhanced_metrics()
    
    print("="*80)
    print("🏆 EDGE-ENHANCED TRAINING COMPLETED")
    print("="*80)
    print(f"📊 Best Dice Score: {metrics_tracker.best_dice:.4f}")
    print(f"📊 Best Epoch: {metrics_tracker.best_epoch}")
    print(f"📁 Results saved in: {experiment_dir}")
    print(f"📈 Enhanced training curves saved")
    
    # Performance summary
    print(f"\n🎯 Performance Summary:")
    print(f"📊 Target: 91% Dice (Paper)")
    print(f"🏆 Achieved: {metrics_tracker.best_dice*100:.2f}% Dice")
    gap = 91 - metrics_tracker.best_dice*100
    print(f"📈 Gap: {gap:.2f} percentage points")
    
    if metrics_tracker.best_dice >= 0.91:
        print("🎉 TARGET ACHIEVED! Edge enhancement successful!")
    elif metrics_tracker.best_dice >= 0.90:
        print("🌟 Excellent performance! Very close to target!")
    else:
        print("📈 Good progress! Consider further edge enhancement tuning.")

if __name__ == "__main__":
    main()
