"""
Enhanced Fine-tuning for DermoMamba: Optimizing for BOTH Dice AND IoU
Multi-objective optimization to achieve highest performance on both metrics
"""
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import random
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root
sys.path.append('d:/Research/DermoMamba')

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from metric.metrics import dice_score, iou_score

class DualMetricLoss(nn.Module):
    """Combined loss function optimizing for both Dice and IoU"""
    def __init__(self, dice_weight=0.4, iou_weight=0.4, bce_weight=0.2, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
    def dice_loss(self, y_pred, y_true):
        """Dice loss calculation"""
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
    
    def iou_loss(self, y_pred, y_true):
        """IoU loss calculation"""
        y_pred = torch.sigmoid(y_pred)
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou
    
    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        iou_loss = self.iou_loss(y_pred, y_true)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
        
        total_loss = (self.dice_weight * dice_loss + 
                     self.iou_weight * iou_loss + 
                     self.bce_weight * bce_loss)
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'iou_loss': iou_loss,
            'bce_loss': bce_loss
        }

class OptimizedDataset(Dataset):
    def __init__(self, data_root, transform=None, split='train'):
        self.transform = transform
        image_dir = os.path.join(data_root, 'train_images')
        mask_dir = os.path.join(data_root, 'train_masks')
        
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        random.seed(42)
        random.shuffle(image_files)
        
        split_idx = int(0.8 * len(image_files))
        self.image_files = image_files[:split_idx] if split == 'train' else image_files[split_idx:]
        self.mask_dir = mask_dir
        
        print(f"📊 {split.upper()} dataset: {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{img_name}_segmentation.png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask.astype(np.float32))
            return transformed['image'], transformed['mask']
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask

class DualMetricModel(pl.LightningModule):
    def __init__(self, checkpoint_path, lr=2e-5, dice_weight=0.4, iou_weight=0.4, bce_weight=0.2):
        super().__init__()
        self.lr = lr
        self.model = OptimizedDermoMamba(n_class=1)
        self.loss_fn = DualMetricLoss(dice_weight, iou_weight, bce_weight)
        
        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remove model. prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict, strict=False)
        print("✅ Loaded pretrained weights with dual-metric optimization")
        
        # Track best metrics
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.best_combined = 0.0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # Ensure mask has same shape as output
        if masks.dim() == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Multi-objective loss
        loss_dict = self.loss_fn(outputs, masks)
        
        # Metrics
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        # Logging
        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train_dice_loss', loss_dict['dice_loss'])
        self.log('train_iou_loss', loss_dict['iou_loss'])
        self.log('train_bce_loss', loss_dict['bce_loss'])
        self.log('train_dice', dice, prog_bar=True)
        self.log('train_iou', iou)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # Ensure mask has same shape as output
        if masks.dim() == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Loss calculation
        loss_dict = self.loss_fn(outputs, masks)
        
        # Metrics
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        # Combined metric for model selection (weighted average)
        combined_metric = 0.6 * dice + 0.4 * iou
        
        # Logging
        self.log('val_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val_dice_loss', loss_dict['dice_loss'])
        self.log('val_iou_loss', loss_dict['iou_loss'])
        self.log('val_bce_loss', loss_dict['bce_loss'])
        self.log('val_dice', dice, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_combined', combined_metric, prog_bar=True)
        
        return {
            'val_loss': loss_dict['total_loss'], 
            'val_dice': dice, 
            'val_iou': iou,
            'val_combined': combined_metric
        }
    
    def on_validation_epoch_end(self):
        # Track best metrics using logged values
        if hasattr(self.trainer, 'logged_metrics'):
            metrics = self.trainer.logged_metrics
            
            if 'val_dice' in metrics:
                current_dice = metrics['val_dice'].item()
                if current_dice > self.best_dice:
                    self.best_dice = current_dice
                    
            if 'val_iou' in metrics:
                current_iou = metrics['val_iou'].item()
                if current_iou > self.best_iou:
                    self.best_iou = current_iou
                    
            if 'val_combined' in metrics:
                current_combined = metrics['val_combined'].item()
                if current_combined > self.best_combined:
                    self.best_combined = current_combined
    
    def configure_optimizers(self):
        # Slightly different optimization for dual metrics
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=self.lr * 0.01
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

def get_enhanced_transforms(image_size=224, is_train=True):
    """Enhanced augmentation strategy for medical images"""
    if is_train:
        return A.Compose([
            # Size and geometric
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.Affine(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            
            # Color and appearance
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise and blur for robustness
            A.GaussianBlur(blur_limit=3, p=0.25),
            A.GaussNoise(var_limit=10.0, p=0.2),
            A.ElasticTransform(alpha=50, sigma=5, p=0.2),
            
            # Medical-specific augmentations
            A.CoarseDropout(max_holes=3, max_height=32, max_width=32, fill_value=0, p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def main():
    print("🎯 DUAL-METRIC OPTIMIZATION: DICE + IoU")
    print("🚀 FINE-TUNING DERMOMAMBA")
    print("🎯 Target: 94%+ Dice AND 92%+ IoU")
    print("Starting from: 92.07% Dice baseline")
    
    # Setup
    data_path = 'd:/Research/DermoMamba/data/ISIC2018_test'
    checkpoint_path = 'd:/Research/DermoMamba/checkpoints/optimized_complete_improved/best_model-v1.ckpt'
    
    # Enhanced data loading
    train_transform = get_enhanced_transforms(224, is_train=True)
    val_transform = get_enhanced_transforms(224, is_train=False)
    
    train_dataset = OptimizedDataset(data_path, train_transform, 'train')
    val_dataset = OptimizedDataset(data_path, val_transform, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
    # Enhanced model with dual-metric optimization
    model = DualMetricModel(
        checkpoint_path, 
        lr=2e-5, 
        dice_weight=0.4,  # 40% dice loss
        iou_weight=0.4,   # 40% iou loss  
        bce_weight=0.2    # 20% bce loss
    )
    
    # Enhanced training setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger(
        save_dir="d:/Research/DermoMamba/tb_logs", 
        name=f"dual_metric_finetune_{timestamp}"
    )
    
    # Multiple checkpoint callbacks for different metrics
    dice_checkpoint = ModelCheckpoint(
        dirpath=f"d:/Research/DermoMamba/experiments/dual_metric_{timestamp}",
        filename="best-dice-{epoch:02d}-{val_dice:.4f}",
        monitor='val_dice',
        mode='max',
        save_top_k=2,
        verbose=True
    )
    
    iou_checkpoint = ModelCheckpoint(
        dirpath=f"d:/Research/DermoMamba/experiments/dual_metric_{timestamp}",
        filename="best-iou-{epoch:02d}-{val_iou:.4f}",
        monitor='val_iou',
        mode='max',
        save_top_k=2,
        verbose=True
    )
    
    combined_checkpoint = ModelCheckpoint(
        dirpath=f"d:/Research/DermoMamba/experiments/dual_metric_{timestamp}",
        filename="best-combined-{epoch:02d}-{val_combined:.4f}",
        monitor='val_combined',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_combined', 
        patience=12, 
        mode='max', 
        min_delta=0.0005,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[dice_checkpoint, iou_checkpoint, combined_checkpoint, early_stop, lr_monitor],
        precision='16-mixed',
        accelerator='gpu',
        devices=1,
        gradient_clip_val=1.0,
        val_check_interval=0.5,
        accumulate_grad_batches=2,  # Effective batch size = 16
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("🔥 Starting dual-metric fine-tuning...")
    print("📈 Optimizing for BOTH Dice AND IoU!")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Results summary
    best_dice = dice_checkpoint.best_model_score
    best_iou = iou_checkpoint.best_model_score
    best_combined = combined_checkpoint.best_model_score
    
    dice_improvement = (best_dice - 0.9207) * 100
    
    print(f"\n🏆 DUAL-METRIC RESULTS:")
    print(f"{'='*50}")
    print(f"📊 Best Dice Score: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"📊 Best IoU Score: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"📊 Best Combined: {best_combined:.4f}")
    print(f"📈 Dice Improvement: {dice_improvement:+.2f} percentage points")
    print(f"{'='*50}")
    
    print(f"\n💾 SAVED MODELS:")
    print(f"🎯 Best Dice: {dice_checkpoint.best_model_path}")
    print(f"🎯 Best IoU: {iou_checkpoint.best_model_path}")
    print(f"🎯 Best Combined: {combined_checkpoint.best_model_path}")
    
    # Performance evaluation
    if best_dice > 0.94 and best_iou > 0.92:
        print("\n🎉 DUAL TARGET ACHIEVED: >94% Dice AND >92% IoU!")
    elif best_dice > 0.94:
        print("\n🎉 DICE TARGET ACHIEVED: >94% Dice!")
    elif best_dice > 0.93:
        print("\n👍 EXCELLENT DICE: >93%!")
    elif best_dice > 0.9207:
        print("\n✅ DICE IMPROVED!")
        
    if best_iou > 0.92:
        print("🎉 IoU TARGET ACHIEVED: >92%!")
    elif best_iou > 0.90:
        print("👍 EXCELLENT IoU: >90%!")

if __name__ == "__main__":
    main()
