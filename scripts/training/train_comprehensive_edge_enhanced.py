#!/usr/bin/env python3
"""
Comprehensive Edge-Enhanced DermoMamba Training
Uses ALL edge enhancement methods together for maximum boundary detection performance
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

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from datasets.isic_dataset import create_data_loaders
from loss.paper_guide_fusion_loss import AdaptiveGuideFusionLoss
from loss.enhanced_boundary_loss import AdvancedBoundaryLoss

class ComprehensiveMetricsTracker:
    """Enhanced metrics tracker for comprehensive edge training"""
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
    
    def plot_comprehensive_curves(self):
        """Create comprehensive training curves"""
        if not self.train_metrics:
            return
            
        plt.figure(figsize=(20, 12))
        epochs = [m['epoch'] + 1 for m in self.train_metrics]
        
        # Main metrics plots
        plt.subplot(3, 4, 1)
        plt.plot(epochs, [m['loss'] for m in self.train_metrics], label='Train', color='blue', alpha=0.7)
        plt.plot(epochs, [m['loss'] for m in self.val_metrics], label='Val', color='red', alpha=0.7)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        plt.plot(epochs, [m['dice'] for m in self.train_metrics], label='Train', color='blue', alpha=0.7)
        plt.plot(epochs, [m['dice'] for m in self.val_metrics], label='Val', color='red', alpha=0.7)
        plt.axhline(y=0.91, color='green', linestyle='--', alpha=0.7, label='Paper Target')
        plt.title('Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 3)
        plt.plot(epochs, [m['iou'] for m in self.train_metrics], label='Train', color='blue', alpha=0.7)
        plt.plot(epochs, [m['iou'] for m in self.val_metrics], label='Val', color='red', alpha=0.7)
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 4)
        plt.plot(epochs, [m.get('learning_rate', 0) for m in self.train_metrics])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True, alpha=0.3)
        
        # Loss component analysis (if available)
        if 'dice_loss' in self.train_metrics[0]:
            plt.subplot(3, 4, 5)
            plt.plot(epochs, [m['dice_loss'] for m in self.train_metrics], label='Dice Loss', alpha=0.7)
            plt.plot(epochs, [m['bce_loss'] for m in self.train_metrics], label='BCE Loss', alpha=0.7)
            plt.plot(epochs, [m.get('boundary_loss', 0) for m in self.train_metrics], label='Boundary Loss', alpha=0.7)
            plt.title('Loss Components (Train)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(3, 4, 6)
        plt.text(0.1, 0.9, f'Best Dice: {self.best_dice:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Best IoU: {self.best_iou:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Best Epoch: {self.best_epoch + 1}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Paper Target: 91.0%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'Gap: {91.0 - self.best_dice*100:.2f}%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, 'Comprehensive Edge Enhancement:', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, '• Unsharp Masking', fontsize=9, transform=plt.gca().transAxes)
        plt.text(0.1, 0.1, '• CLAHE + Sobel + Boundary Opt', fontsize=9, transform=plt.gca().transAxes)
        plt.title('Training Summary')
        plt.axis('off')
        
        # Training progress over time
        if len(epochs) > 10:
            plt.subplot(3, 4, 7)
            recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
            recent_dice = [m['dice'] for m in self.val_metrics[-20:]] if len(self.val_metrics) > 20 else [m['dice'] for m in self.val_metrics]
            plt.plot(recent_epochs, recent_dice, 'o-', color='red', alpha=0.7)
            plt.title('Recent Validation Dice')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.grid(True, alpha=0.3)
        
        # Model convergence analysis
        plt.subplot(3, 4, 8)
        if len(epochs) > 5:
            dice_diff = np.diff([m['dice'] for m in self.val_metrics])
            plt.plot(epochs[1:], dice_diff, alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.title('Dice Improvement Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Change')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'comprehensive_training_curves.png'), 
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

def train_epoch_comprehensive(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Comprehensive training epoch with all edge enhancements"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # Track comprehensive enhancement effects
    num_batches = len(train_loader)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            
            # Handle different loss types
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

def validate_comprehensive(model, val_loader, criterion, device, epoch):
    """Comprehensive validation with all edge enhancements"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                
                # Handle different loss types
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
    parser = argparse.ArgumentParser(description='Comprehensive Edge-Enhanced DermoMamba Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--edge_method', type=str, default='comprehensive',
                       choices=['comprehensive', 'boundary_optimized', 'unsharp_mask', 'clahe_edges', 'sobel'],
                       help='Edge enhancement method')
    parser.add_argument('--loss_type', type=str, default='adaptive_guide_fusion',
                       choices=['guide_fusion', 'adaptive_guide_fusion', 'advanced_boundary'],
                       help='Loss function type')
    args = parser.parse_args()

    print("="*80)
    print("🚀 COMPREHENSIVE EDGE-ENHANCED DERMOMAMBA TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"comprehensive_edge_enhanced_{args.loss_type}_{timestamp}"
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Save configuration
    config = {
        'edge_enhancement': 'comprehensive_all_methods',
        'loss_type': args.loss_type,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'device': str(device),
        'model': 'OptimizedDermoMamba',
        'comprehensive_edge_methods': [
            'unsharp_masking',
            'clahe_contrast',
            'sobel_edges', 
            'boundary_optimization',
            'multi_scale_features'
        ],
        'target_dice': 0.91
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders with comprehensive edge enhancement
    print("🔄 Loading data with COMPREHENSIVE edge enhancement...")
    print("   • Unsharp Masking ✓")
    print("   • CLAHE Contrast ✓") 
    print("   • Sobel Edge Detection ✓")
    print("   • Boundary Optimization ✓")
    print("   • Multi-scale Features ✓")
    
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
    
    # Initialize loss function
    if args.loss_type == 'adaptive_guide_fusion':
        criterion = AdaptiveGuideFusionLoss(
            dice_weight=1.0,
            bce_weight=0.5,
            boundary_weight=2.0,  # Higher weight for enhanced boundaries
            attention_weight=1.5
        )
    elif args.loss_type == 'advanced_boundary':
        criterion = AdvancedBoundaryLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    criterion = criterion.to(device)
    print(f"✅ Loss function: {args.loss_type}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    print("✅ Setup complete, starting comprehensive enhanced training...")
    print("🎯 Target: 91% Dice Score (Paper)")
    print("="*80)
    
    # Initialize metrics tracker
    metrics_tracker = ComprehensiveMetricsTracker(experiment_dir)
    
    # Training loop
    best_dice = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train epoch
        train_metrics = train_epoch_comprehensive(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate epoch  
        val_metrics = validate_comprehensive(model, val_loader, criterion, device, epoch)
        
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
        print(f"  Comprehensive Edge Enhancement: ALL METHODS")
        
        # Save best model
        if is_best:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou'],
                'train_dice': train_metrics['dice'],
                'train_iou': train_metrics['iou']
            }
            
            torch.save(checkpoint, os.path.join(experiment_dir, 'checkpoints', 'best_model.ckpt'))
            print(f"  💾 New best model saved! Dice: {val_metrics['dice']:.4f}")
            
            # Check if we reached the paper's target
            if val_metrics['dice'] >= 0.91:
                print(f"  🎯 REACHED PAPER TARGET! Dice: {val_metrics['dice']:.4f} >= 91%")
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
                'val_iou': val_metrics['iou']
            }
            torch.save(checkpoint, os.path.join(experiment_dir, 'checkpoints', f'model_epoch_{epoch+1}.ckpt'))
        
        # Save metrics and plot curves every few epochs
        if (epoch + 1) % 5 == 0:
            metrics_tracker.save_metrics()
            metrics_tracker.plot_comprehensive_curves()
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping after {epoch+1} epochs (patience: {patience})")
            break
    
    # Final save
    metrics_tracker.save_metrics()
    metrics_tracker.plot_comprehensive_curves()
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_dice': val_metrics['dice'],
        'val_iou': val_metrics['iou']
    }
    torch.save(final_checkpoint, os.path.join(experiment_dir, 'checkpoints', 'final_model.ckpt'))
    
    print("="*80)
    print("🏆 COMPREHENSIVE TRAINING COMPLETED")
    print("="*80)
    print(f"📊 Best Dice Score: {metrics_tracker.best_dice:.4f}")
    print(f"📊 Best IoU Score: {metrics_tracker.best_iou:.4f}")
    print(f"📊 Best Epoch: {metrics_tracker.best_epoch + 1}")
    print(f"📁 Results saved in: {experiment_dir}")
    print(f"📈 Training curves saved")
    print()
    print(f"🎯 Target: 91% Dice (Paper)")
    print(f"🏆 Achieved: {metrics_tracker.best_dice*100:.2f}% Dice")
    
    gap = 91.0 - metrics_tracker.best_dice*100
    if gap <= 0:
        print(f"✅ TARGET EXCEEDED by {abs(gap):.2f} percentage points!")
    else:
        print(f"📈 Gap: {gap:.2f} percentage points")
    
    print("🔬 Comprehensive Edge Enhancement Used:")
    print("  • Unsharp Masking")
    print("  • CLAHE Contrast Enhancement") 
    print("  • Sobel Edge Detection")
    print("  • Boundary Optimization")
    print("  • Multi-scale Features")

if __name__ == "__main__":
    main()
