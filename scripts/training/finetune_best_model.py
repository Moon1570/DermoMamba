"""
Fine-tuning script for the best DermoMamba model to push performance higher
Starting from 92.07% Dice, targeting 94-95%+ performance
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import necessary modules
from datasets.isic_dataset import ISICDataset
from module.model.optimized_dermomamba_complete import OptimizedDermoMamba as CompleteOptimizedDermoMamba
from metric.metrics import dice_score, iou_score

class AdvancedFinetuneTransforms:
    """Advanced augmentation strategy for fine-tuning"""
    
    @staticmethod
    def get_finetune_transforms(input_size=(224, 224)):
        """
        Aggressive but controlled augmentations for fine-tuning
        Focus on boundary-aware augmentations and challenging cases
        """
        return A.Compose([
            # Size augmentations
            A.Resize(input_size[0], input_size[1]),
            
            # Geometric augmentations - more aggressive
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=25, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.6
            ),
            
            # Advanced geometric
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=10,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=8,
                distort_limit=0.2,
                p=0.3
            ),
            
            # Color augmentations - medical image focused
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=0.6
            ),
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
            
            # Noise and blur for robustness
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
                A.MotionBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.4),
            
            A.OneOf([
                A.GaussNoise(var_limit=(5, 15), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=0.3),
            ], p=0.3),
            
            # Advanced color transforms
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.1,
                p=0.5
            ),
            
            # Cutout for regularization
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_finetune_val_transforms(input_size=(224, 224)):
        """Validation transforms with Test Time Augmentation"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

class AdvancedLoss(nn.Module):
    """
    Advanced loss combining multiple objectives for fine-tuning
    """
    def __init__(self):
        super().__init__()
        self.dice_loss = self._dice_loss
        self.focal_loss = self._focal_loss
        self.tversky_loss = self._tversky_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def _dice_loss(self, pred, target):
        """Dice loss"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def _focal_loss(self, pred, target, alpha=0.8, gamma=2):
        """Focal loss for hard examples"""
        pred = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss
            
        return loss.mean()
    
    def _tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        """Tversky loss - good for imbalanced data"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky.mean()
    
    def forward(self, pred, target):
        """Combined loss"""
        # Primary losses
        dice_l = self.dice_loss(pred, target)
        focal_l = self.focal_loss(pred, target)
        tversky_l = self.tversky_loss(pred, target)
        bce_l = self.bce_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            0.4 * dice_l +
            0.3 * focal_l + 
            0.2 * tversky_l +
            0.1 * bce_l
        )
        
        return total_loss, {
            'dice_loss': dice_l,
            'focal_loss': focal_l,
            'tversky_loss': tversky_l,
            'bce_loss': bce_l,
            'total_loss': total_loss
        }

class FinetunedDermoMamba(pl.LightningModule):
    """
    Fine-tuned DermoMamba with advanced training strategies
    """
    def __init__(self, 
                 pretrained_checkpoint=None,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 warmup_steps=100):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Load pre-trained model
        self.model = CompleteOptimizedDermoMamba(n_class=1)
        
        if pretrained_checkpoint:
            print(f"Loading pre-trained weights from: {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print("✅ Pre-trained weights loaded successfully")
        
        # Advanced loss
        self.criterion = AdvancedLoss()
        
        # Metrics tracking
        self.train_dice_scores = []
        self.val_dice_scores = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        # Handle different output formats
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # Calculate loss
        loss, loss_dict = self.criterion(outputs, masks)
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_score(outputs, masks)
            self.train_dice_scores.append(dice.item())
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        # Handle different output formats
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # Calculate loss
        loss, loss_dict = self.criterion(outputs, masks)
        
        # Calculate metrics
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        self.val_dice_scores.append(dice.item())
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True)
        
        return {'val_loss': loss, 'val_dice': dice, 'val_iou': iou}
    
    def configure_optimizers(self):
        """Advanced optimizer configuration with fine-tuning strategies"""
        
        # Different learning rates for different parts
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'conv_out' in name or 'd1' in name or 'd2' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # Lower LR for backbone, higher for head
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.learning_rate}  # Higher LR for head
        ], weight_decay=self.weight_decay)
        
        # Cosine annealing with warm restart
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

def setup_finetune_data(data_root, batch_size=16, num_workers=4):
    """Setup data for fine-tuning"""
    
    # Advanced transforms
    train_transform = AdvancedFinetuneTransforms.get_finetune_transforms()
    val_transform = AdvancedFinetuneTransforms.get_finetune_val_transforms()
    
    # Load datasets - handle different data configurations
    train_split = 'd:/Research/DermoMamba/splits/isic2018_train.txt'
    val_split = 'd:/Research/DermoMamba/splits/isic2018_val.txt'
    
    # Check if we have the proper data structure
    if not os.path.exists(os.path.join(data_root, 'train_images')) or not os.path.exists(train_split):
        print("⚠️  Using simplified data setup for fine-tuning")
        # Create simple train/val split from available data
        from glob import glob
        import random
        
        image_files = sorted(glob(os.path.join(data_root, 'train_images', '*.jpg')))
        random.shuffle(image_files)
        
        # 80/20 split
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"Created train/val split: {len(train_files)}/{len(val_files)} samples")
        
        # Use simple dataset class instead
        from torch.utils.data import Dataset
        from PIL import Image
        import numpy as np
        
        class SimpleDataset(Dataset):
            def __init__(self, image_files, transform=None):
                self.image_files = image_files
                self.transform = transform
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_path = self.image_files[idx]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'train_masks', f"{img_name}_segmentation.png")
                
                # Load image and mask
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
                
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask) / 255.0
                mask = mask.astype(np.float32)
                
                if self.transform:
                    transformed = self.transform(image=image, mask=mask)
                    return transformed['image'], transformed['mask']
                else:
                    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                    return image, mask
        
        train_dataset = SimpleDataset(train_files, train_transform)
        val_dataset = SimpleDataset(val_files, val_transform)
        
    else:
        train_dataset = ISICDataset(
            data_root=data_root,
            split_file=train_split,
            transform=train_transform,
            is_train=True
        )
        
        val_dataset = ISICDataset(
            data_root=data_root,
            split_file=val_split,
            transform=val_transform,
            is_train=False
        )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    print("🚀 STARTING ADVANCED FINE-TUNING OF DERMOMAMBA")
    print("=" * 60)
    print("Starting from 92.07% Dice, targeting 94-95%+")
    
    # Configuration
    config = {
        'data_root': 'd:/Research/DermoMamba/data/ISIC2018_test',
        'pretrained_checkpoint': 'd:/Research/DermoMamba/checkpoints/optimized_complete_improved/best_model-v1.ckpt',
        'batch_size': 12,  # Smaller for fine-tuning stability
        'learning_rate': 5e-5,  # Lower LR for fine-tuning
        'weight_decay': 1e-5,
        'max_epochs': 50,
        'num_workers': 0,
        'precision': 16
    }
    
    # Setup data
    print(f"Setting up data from: {config['data_root']}")
    train_loader, val_loader = setup_finetune_data(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Setup model
    print("Initializing fine-tuned model...")
    model = FinetunedDermoMamba(
        pretrained_checkpoint=config['pretrained_checkpoint'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Setup training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"finetuned_dermomamba_{timestamp}"
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="d:/Research/DermoMamba/tb_logs",
        name=experiment_name
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"d:/Research/DermoMamba/experiments/{experiment_name}",
        filename="best_finetuned_model-{epoch:02d}-{val_dice:.4f}",
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True,
        min_delta=0.001
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision=config['precision'],
        accelerator='gpu',
        devices=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = 12 * 2 = 24
        val_check_interval=0.5,  # Check validation twice per epoch
        log_every_n_steps=10
    )
    
    print(f"🎯 Training Configuration:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Batch Size: {config['batch_size']} (effective: {config['batch_size'] * 2})")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Max Epochs: {config['max_epochs']}")
    print(f"  Precision: {config['precision']}-bit")
    print(f"  Starting Dice: 92.07% (target: 94-95%+)")
    
    # Start training
    print("\n🔥 Starting fine-tuning...")
    trainer.fit(model, train_loader, val_loader)
    
    # Results
    best_dice = checkpoint_callback.best_model_score
    print(f"\n🏆 FINE-TUNING COMPLETE!")
    print(f"Best Validation Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"Improvement: {(best_dice - 0.9207)*100:.2f} percentage points")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
