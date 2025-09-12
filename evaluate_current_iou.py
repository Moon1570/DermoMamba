"""
Quick IoU evaluation of our current best model (94.50% Dice)
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root
sys.path.append('d:/Research/DermoMamba')

from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
from metric.metrics import dice_score, iou_score

class SimpleDataset(Dataset):
    def __init__(self, data_root, transform=None, split='val'):
        self.transform = transform
        
        if split == 'val':
            image_dir = os.path.join(data_root, 'val_images')
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
            self.image_files = image_files
        else:
            # Use train split for validation
            image_dir = os.path.join(data_root, 'train_images')
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
            random.seed(42)
            random.shuffle(image_files)
            
            # Use validation split (last 20%)
            split_idx = int(0.8 * len(image_files))
            self.image_files = image_files[split_idx:]
            
        mask_dir = os.path.join(data_root, 'train_masks')
        self.mask_dir = mask_dir
        
        print(f"Evaluation dataset: {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{img_name}_segmentation.png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.float()

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def evaluate_current_model():
    print("🔍 EVALUATING CURRENT BEST MODEL IoU PERFORMANCE")
    print("Model: 94.50% Dice (from fine-tuning)")
    
    # Load model
    model_path = 'experiments/finetune_20250911_014557/best-epoch=07-val_dice=0.9450.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OptimizedDermoMamba()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Dataset
    data_root = 'data/ISIC2018_proc'
    transform = get_val_transforms()
    dataset = SimpleDataset(data_root, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Evaluation
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    
    print("🧮 Computing metrics...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Ensure mask has correct shape
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Calculate metrics
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            
            total_dice += dice * images.size(0)
            total_iou += iou * images.size(0)
            total_samples += images.size(0)
            
            print(f"Batch {batch_idx+1}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # Final averages
    avg_dice = total_dice / total_samples
    avg_iou = total_iou / total_samples
    
    print("\n" + "="*50)
    print("📊 CURRENT MODEL PERFORMANCE:")
    print(f"🎯 Average Dice Score: {avg_dice:.4f} ({avg_dice*100:.2f}%)")
    print(f"📈 Average IoU Score:  {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print("="*50)
    
    print(f"\n🎯 IoU Performance Analysis:")
    if avg_iou >= 0.89:
        print("✅ IoU target (89%+) already achieved!")
    else:
        gap = 0.89 - avg_iou
        print(f"📈 IoU gap to 89%: {gap:.4f} ({gap*100:.2f} percentage points)")
        print("🔥 Multi-objective optimization recommended!")
    
    return avg_dice, avg_iou

if __name__ == "__main__":
    evaluate_current_model()
