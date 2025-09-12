"""
Working fine-tuning script for DermoMamba
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

class FinetuneDataset(Dataset):
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
        
        print(f"{split} dataset: {len(self.image_files)} samples")
    
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

class FinetuneModel(pl.LightningModule):
    def __init__(self, checkpoint_path, lr=2e-5):
        super().__init__()
        self.lr = lr
        self.model = OptimizedDermoMamba(n_class=1)
        
        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remove model. prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict, strict=False)
        print("✅ Loaded pretrained weights")
    
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
        
        # Combined loss
        dice_loss = 1 - dice_score(outputs, masks)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        loss = 0.7 * dice_loss + 0.3 * bce_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice', dice_score(outputs, masks), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # Ensure mask has same shape as output
        if masks.dim() == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        dice_loss = 1 - dice_score(outputs, masks)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        loss = 0.7 * dice_loss + 0.3 * bce_loss
        
        dice = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        self.log('val_iou', iou)
        
        return {'val_loss': loss, 'val_dice': dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=self.lr * 0.01)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def get_transforms(image_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.2),
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
    print("🚀 FINE-TUNING DERMOMAMBA")
    print("Starting from 92.07% Dice")
    
    # Setup
    data_path = 'd:/Research/DermoMamba/data/ISIC2018_test'
    checkpoint_path = 'd:/Research/DermoMamba/checkpoints/optimized_complete_improved/best_model-v1.ckpt'
    
    # Data
    train_transform = get_transforms(224, is_train=True)
    val_transform = get_transforms(224, is_train=False)
    
    train_dataset = FinetuneDataset(data_path, train_transform, 'train')
    val_dataset = FinetuneDataset(data_path, val_transform, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Model
    model = FinetuneModel(checkpoint_path, lr=2e-5)
    
    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger(save_dir="d:/Research/DermoMamba/tb_logs", name=f"finetune_{timestamp}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"d:/Research/DermoMamba/experiments/finetune_{timestamp}",
        filename="best-{epoch:02d}-{val_dice:.4f}",
        monitor='val_dice',
        mode='max',
        save_top_k=2,
        verbose=True
    )
    
    early_stop = EarlyStopping(monitor='val_dice', patience=8, mode='max', min_delta=0.001)
    
    trainer = pl.Trainer(
        max_epochs=25,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
        precision='16-mixed',
        accelerator='gpu',
        devices=1,
        gradient_clip_val=0.5,
        val_check_interval=0.5
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("🔥 Starting fine-tuning...")
    
    trainer.fit(model, train_loader, val_loader)
    
    best_dice = checkpoint_callback.best_model_score
    improvement = (best_dice - 0.9207) * 100
    
    print(f"\n🏆 RESULTS:")
    print(f"Best Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"Improvement: {improvement:+.2f} percentage points")
    print(f"Model: {checkpoint_callback.best_model_path}")
    
    if best_dice > 0.94:
        print("🎉 TARGET ACHIEVED: >94% Dice!")
    elif best_dice > 0.93:
        print("👍 EXCELLENT: >93% Dice!")
    elif best_dice > 0.9207:
        print("✅ IMPROVED!")

if __name__ == "__main__":
    main()
