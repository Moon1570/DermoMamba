"""
DermoMamba Training with Guide Fusion Loss
Advanced training script using boundary-guided loss functions
to achieve superior Dice and IoU scores for skin lesion segmentation.

Features:
- Guide Fusion Loss from the paper
- Enhanced boundary-aware losses  
- Multi-scale training
- Adaptive loss weighting
- Comprehensive metrics tracking
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will be skipped")
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.append('.')
sys.path.append('../..')

def setup_gpu():
    """Setup GPU and check availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear cache
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("❌ GPU not available, using CPU")
    
    return device

def create_experiment_dir(base_name="guide_fusion_experiment"):
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{base_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    return exp_dir

class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.train_metrics = []
        self.val_metrics = []
        self.best_dice = 0
        self.best_iou = 0
        
    def log_epoch(self, epoch, train_metrics, val_metrics):
        """Log metrics for an epoch"""
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Update best scores
        if val_metrics['dice'] > self.best_dice:
            self.best_dice = val_metrics['dice']
        if val_metrics['iou'] > self.best_iou:
            self.best_iou = val_metrics['iou']
        
        # Save metrics
        with open(f"{self.exp_dir}/logs/train_metrics.json", 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        with open(f"{self.exp_dir}/logs/val_metrics.json", 'w') as f:
            json.dump(self.val_metrics, f, indent=2)
    
    def plot_metrics(self):
        """Plot training curves"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available, skipping plots")
            return
            
        epochs = [m['epoch'] for m in self.train_metrics]
        
        # Plot losses
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(epochs, [m['loss'] for m in self.train_metrics], label='Train', color='blue')
        plt.plot(epochs, [m['loss'] for m in self.val_metrics], label='Val', color='red')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(epochs, [m['dice'] for m in self.train_metrics], label='Train', color='blue')
        plt.plot(epochs, [m['dice'] for m in self.val_metrics], label='Val', color='red')
        plt.title('Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(epochs, [m['iou'] for m in self.train_metrics], label='Train', color='blue')
        plt.plot(epochs, [m['iou'] for m in self.val_metrics], label='Val', color='red')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        # Plot individual loss components if available
        if 'dice_loss' in self.train_metrics[0]:
            plt.subplot(2, 3, 4)
            plt.plot(epochs, [m['dice_loss'] for m in self.train_metrics], label='Dice Loss')
            plt.plot(epochs, [m['bce_loss'] for m in self.train_metrics], label='BCE Loss')
            plt.plot(epochs, [m.get('boundary_loss', 0) for m in self.train_metrics], label='Boundary Loss')
            plt.title('Loss Components (Train)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(epochs, [m['learning_rate'] for m in self.train_metrics])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f'Best Dice: {self.best_dice:.4f}', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Best IoU: {self.best_iou:.4f}', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'Final Dice: {self.val_metrics[-1]["dice"]:.4f}', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Final IoU: {self.val_metrics[-1]["iou"]:.4f}', fontsize=14, transform=plt.gca().transAxes)
        plt.title('Final Results')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/plots/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

def dice_score(pred, target, smooth=1e-6):
    """Compute Dice score"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def iou_score(pred, target, smooth=1e-6):
    """Compute IoU score"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, metrics_tracker):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    loss_components = {'dice_loss': 0, 'bce_loss': 0, 'boundary_loss': 0, 'attention_dice': 0}
    
    num_batches = len(loader)
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            
            # Handle different loss function returns
            if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                # Advanced loss functions that return loss dict
                try:
                    loss, loss_dict = criterion(outputs, masks)
                    for key, value in loss_dict.items():
                        if key in loss_components:
                            loss_components[key] += value
                except:
                    loss = criterion(outputs, masks)
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
        
        # Progress logging
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, "
                  f"Dice: {dice.item():.4f}, IoU: {iou.item():.4f}")
    
    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    # Add loss components if available
    for key, value in loss_components.items():
        if value > 0:
            avg_metrics[key] = value / num_batches
    
    return avg_metrics

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    num_batches = len(loader)
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            
            with autocast():
                outputs = model(images)
                
                # Handle different loss function returns
                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    try:
                        loss, _ = criterion(outputs, masks)
                    except:
                        loss = criterion(outputs, masks)
                else:
                    loss = criterion(outputs, masks)
            
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
    print("="*80)
    print("🔬 DERMOMAMBA GUIDE FUSION LOSS TRAINING")
    print("="*80)
    
    # Setup
    device = setup_gpu()
    exp_dir = create_experiment_dir("guide_fusion_boundary_enhanced")
    
    print(f"📁 Experiment directory: {exp_dir}")
    
    # Import after path setup
    from datasets.isic_dataset import ISICDataset
    from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
    from loss.paper_guide_fusion_loss import create_guide_fusion_loss
    from loss.enhanced_boundary_loss import create_boundary_loss
    
    # Configuration
    config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'weight_decay': 1e-5,
        'loss_type': 'guide_fusion',  # 'guide_fusion', 'boundary_enhanced', 'advanced_boundary'
        'model_type': 'optimized_complete',
        'image_size': 384,
        'early_stopping_patience': 20,
        'lr_scheduler': 'cosine'
    }
    
    # Save config
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Dataset paths
    data_root = "data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    if not os.path.exists(data_root):
        print(f"❌ Data not found at {data_root}")
        return
    
    # Create datasets
    train_dataset = ISICDataset(
        data_root=data_root,
        split_file=train_split,
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split_file=val_split,
        is_train=False
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Create model
    model = OptimizedDermoMamba(n_class=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Create loss function
    if config['loss_type'] == 'guide_fusion':
        criterion = create_guide_fusion_loss('adaptive',
                                           dice_weight=1.0,
                                           bce_weight=1.0,
                                           boundary_weight=2.5,
                                           attention_weight=1.8,
                                           multiscale_weight=0.8).to(device)
    elif config['loss_type'] == 'boundary_enhanced':
        criterion = create_boundary_loss('advanced',
                                       dice_weight=1.0,
                                       edge_weight=2.5,
                                       contour_weight=2.0,
                                       topology_weight=1.2,
                                       consistency_weight=0.8).to(device)
    else:
        # Fallback to standard loss
        from loss.loss import DiceLoss
        criterion = DiceLoss().to(device)
    
    print(f"✅ Loss function: {config['loss_type']}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    
    # Create scheduler
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(exp_dir)
    
    print("✅ Setup complete, starting training...")
    print("="*80)
    
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Update adaptive loss weights if applicable
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                   scaler, device, epoch, metrics_tracker)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if config['lr_scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_metrics['dice'])
        
        # Log metrics
        metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'config': config
            }
            
            torch.save(checkpoint, f"{exp_dir}/checkpoints/best_model.ckpt")
            print(f"  💾 New best model saved! Dice: {best_dice:.4f}")
            
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⏹️ Early stopping after {epoch+1} epochs")
            break
        
        # Save regular checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, f"{exp_dir}/checkpoints/model_epoch_{epoch+1}.ckpt")
    
    # Final results
    print("\n" + "="*80)
    print("🏆 TRAINING COMPLETED")
    print("="*80)
    print(f"📊 Best Dice Score: {metrics_tracker.best_dice:.4f}")
    print(f"📊 Best IoU Score: {metrics_tracker.best_iou:.4f}")
    print(f"📁 Results saved in: {exp_dir}")
    
    # Plot final results
    metrics_tracker.plot_metrics()
    print("📈 Training curves saved")
    
    # Final model save
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_dice': metrics_tracker.best_dice,
        'best_iou': metrics_tracker.best_iou,
        'config': config,
        'final_metrics': val_metrics
    }
    torch.save(final_checkpoint, f"{exp_dir}/checkpoints/final_model.ckpt")
    
    print(f"\n🎯 Target: 91% Dice (Paper)")
    print(f"🏆 Achieved: {metrics_tracker.best_dice*100:.2f}% Dice")
    print(f"📈 Gap: {(0.91 - metrics_tracker.best_dice)*100:.2f} percentage points")
    
    if metrics_tracker.best_dice >= 0.91:
        print("🎉 PAPER PERFORMANCE ACHIEVED! 🎉")
    elif metrics_tracker.best_dice >= 0.90:
        print("🥉 EXCELLENT PERFORMANCE! Very close to paper.")
    elif metrics_tracker.best_dice >= 0.89:
        print("🥈 GREAT PERFORMANCE! Close to paper target.")

if __name__ == "__main__":
    main()
