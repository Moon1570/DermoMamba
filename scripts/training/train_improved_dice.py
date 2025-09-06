"""
Improved training script with better dice score handling
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append('.')

from datasets.isic_dataset import ISICDataset, get_train_transforms, get_val_transforms
from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from loss.loss import DiceLoss, calc_loss, bce_tversky_loss
from metric.metrics import dice_score, iou_score

class ImprovedDiceLoss(nn.Module):
    """Improved Dice Loss with better numerical stability"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # Dice coefficient with smoothing
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice

class CombinedLoss(nn.Module):
    """Combined loss with better weighting for segmentation"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for BCE
        self.beta = beta    # Weight for Dice
        self.dice_loss = ImprovedDiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Combined loss
        return self.alpha * bce + self.beta * dice

def improved_dice_score(pred, target, smooth=1e-6):
    """Improved dice score calculation"""
    pred = torch.sigmoid(pred)
    
    # Apply threshold
    pred_binary = (pred > 0.5).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate dice
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def improved_iou_score(pred, target, smooth=1e-6):
    """Improved IoU score calculation"""
    pred = torch.sigmoid(pred)
    
    # Apply threshold
    pred_binary = (pred > 0.5).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

class ImprovedDermoMambaLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = OptimizedDermoMamba(n_class=1)
        
        # Improved loss function
        self.combined_loss = CombinedLoss(alpha=0.3, beta=0.7, smooth=1.0)
        
        # Improved metrics
        self.dice_metric = improved_dice_score
        self.iou_metric = improved_iou_score
        
        # Learning rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def forward(self, x):
        return self.model(x)  # Don't apply sigmoid here - loss function will handle it
        
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        
        # Ensure mask dimensions match output
        if len(masks.shape) == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Calculate loss
        loss = self.combined_loss(outputs, masks)
        
        # Calculate metrics (with sigmoid applied in metric functions)
        dice = self.dice_metric(outputs, masks)
        iou = self.iou_metric(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_dice', dice, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_iou', iou, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        
        # Ensure mask dimensions match output
        if len(masks.shape) == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Calculate loss
        loss = self.combined_loss(outputs, masks)
        
        # Calculate metrics
        dice = self.dice_metric(outputs, masks)
        iou = self.iou_metric(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_dice', dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_iou', iou, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use AdamW with lower learning rate for better stability
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use ReduceLROnPlateau for adaptive learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'frequency': 1
            }
        }

def main():
    print("="*70)
    print("Training Improved DermoMamba with Enhanced Dice Score")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Data paths
    data_root = "data/ISIC2018_proc"
    
    # Create datasets with proper transforms
    train_dataset = ISICDataset(
        data_root=data_root,
        split_file='splits/isic2018_train.txt',
        transform=get_train_transforms((256, 256)),
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split_file='splits/isic2018_val.txt', 
        transform=get_val_transforms((256, 256)),
        is_train=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create improved model
    model = ImprovedDermoMambaLightning(
        learning_rate=5e-5,  # Lower learning rate
        weight_decay=1e-5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/optimized_complete_improved',
        filename='best_model',
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_dice',
        patience=20,  # Increased patience
        mode='max',
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger('tb_logs', name='improved_complete_dermomamba')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    print(f"\nImproved Training Configuration:")
    print(f"- Batch size: 4 (effective: 8 with grad accumulation)")
    print(f"- Learning rate: 5e-5 (reduced for stability)")
    print(f"- Loss: Combined BCE (30%) + Dice (70%)")
    print(f"- Scheduler: ReduceLROnPlateau")
    print(f"- Gradient clipping: 1.0")
    print(f"- Early stopping patience: 20")
    
    print(f"\nDice Score Improvements:")
    print(f"- Improved numerical stability with smoothing")
    print(f"- Better loss weighting (70% dice, 30% BCE)")
    print(f"- Proper thresholding in metric calculation")
    print(f"- Lower learning rate for stable convergence")
    
    # Start training
    print(f"\n🚀 Starting improved training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and test
    best_model_path = checkpoint_callback.best_model_path
    print(f"\n🏆 Best model saved at: {best_model_path}")
    
    if best_model_path:
        # Load best model for final evaluation
        best_model = ImprovedDermoMambaLightning.load_from_checkpoint(best_model_path)
        trainer.validate(best_model, val_loader)
        
        print(f"\n✅ Improved training completed successfully!")
        print(f"✅ Best validation dice: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()
