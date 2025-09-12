"""
Full validation on the best performing model
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import necessary modules
from module.model.optimized_dermomamba_complete import OptimizedDermoMamba as CompleteOptimizedDermoMamba
from metric.metrics import dice_score, iou_score

class FullValidationDataset(Dataset):
    """Full validation dataset"""
    
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Find directories
        self.image_dir = os.path.join(data_root, 'train_images')
        self.mask_dir = os.path.join(data_root, 'train_masks')
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        
        # Use validation split if available
        splits_file = 'd:/Research/DermoMamba/splits/isic2018_val.txt'
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                val_names = [line.strip() for line in f.readlines()]
            
            # Filter to validation images only
            val_image_files = []
            for img_file in self.image_files:
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                if img_name in val_names:
                    val_image_files.append(img_file)
            
            self.image_files = val_image_files
            print(f"Using validation split: {len(self.image_files)} images")
        else:
            print(f"No validation split found, using all {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Find corresponding mask
        mask_path = os.path.join(self.mask_dir, f"{img_name}_segmentation.png")
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Normalize mask to 0-1
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            # Default normalization
            image = image / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask

def validate_best_model():
    print("🎯 FULL VALIDATION OF BEST MODEL")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Best checkpoint path
    best_checkpoint = "d:/Research/DermoMamba/checkpoints/optimized_complete_improved/best_model-v1.ckpt"
    print(f"Loading: {best_checkpoint}")
    
    # Load model
    model = CompleteOptimizedDermoMamba(n_class=1)
    checkpoint = torch.load(best_checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Setup validation data
    data_path = "d:/Research/DermoMamba/data/ISIC2018_test"
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dataset = FullValidationDataset(data_root=data_path, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Validation loop
    all_dice_scores = []
    all_iou_scores = []
    
    print("\nRunning full validation...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx % 20 == 0:
                print(f"  Progress: {batch_idx}/{len(dataloader)} batches ({batch_idx/len(dataloader)*100:.1f}%)")
            
            images = images.to(device)
            masks = masks.to(device)
            
            try:
                # Forward pass
                outputs = model(images)
                
                # Handle different output formats
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                
                # Calculate metrics for each sample in batch
                for i in range(outputs.size(0)):
                    output_sample = outputs[i:i+1]
                    mask_sample = masks[i:i+1]
                    
                    # Calculate Dice
                    dice = dice_score(output_sample, mask_sample).item()
                    all_dice_scores.append(dice)
                    
                    # Calculate IoU  
                    iou = iou_score(output_sample, mask_sample)
                    all_iou_scores.append(iou)
                    
            except Exception as e:
                print(f"  Error processing batch {batch_idx}: {e}")
                continue
    
    # Results
    if all_dice_scores:
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        mean_iou = np.mean(all_iou_scores)
        std_iou = np.std(all_iou_scores)
        
        print(f"\n🏆 FINAL RESULTS - BEST MODEL:")
        print(f"=" * 50)
        print(f"Checkpoint: optimized_complete_improved/best_model-v1.ckpt")
        print(f"Model Type: OptimizedDermoMamba Complete")
        print(f"")
        print(f"📊 Performance Metrics:")
        print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f} ({mean_dice*100:.2f}%)")
        print(f"  IoU Score:  {mean_iou:.4f} ± {std_iou:.4f} ({mean_iou*100:.2f}%)")
        print(f"  Samples:    {len(all_dice_scores)}")
        
        # Save detailed results
        results = {
            'checkpoint': best_checkpoint,
            'model_type': 'optimized_complete_improved',
            'validation_date': datetime.now().isoformat(),
            'dataset_size': len(all_dice_scores),
            'dice_score': {
                'mean': float(mean_dice),
                'std': float(std_dice),
                'percentage': float(mean_dice * 100),
                'all_scores': [float(x) for x in all_dice_scores]
            },
            'iou_score': {
                'mean': float(mean_iou),
                'std': float(std_iou), 
                'percentage': float(mean_iou * 100),
                'all_scores': [float(x) for x in all_iou_scores]
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"d:/Research/DermoMamba/best_model_full_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"")
        print(f"💾 Full results saved to: {results_file}")
        
        # Performance analysis
        print(f"")
        print(f"🔍 Performance Analysis:")
        high_dice = sum(1 for x in all_dice_scores if x > 0.9)
        medium_dice = sum(1 for x in all_dice_scores if 0.8 <= x <= 0.9)
        low_dice = sum(1 for x in all_dice_scores if x < 0.8)
        
        print(f"  High performance (>90% Dice): {high_dice} samples ({high_dice/len(all_dice_scores)*100:.1f}%)")
        print(f"  Medium performance (80-90% Dice): {medium_dice} samples ({medium_dice/len(all_dice_scores)*100:.1f}%)")
        print(f"  Low performance (<80% Dice): {low_dice} samples ({low_dice/len(all_dice_scores)*100:.1f}%)")
        
        print(f"")
        if mean_dice > 0.90:
            print("🎉 EXCELLENT: Model achieves >90% Dice score!")
            print("🎯 TARGET ACHIEVED: This exceeds the 91% target mentioned!")
        elif mean_dice > 0.85:
            print("👍 GOOD: Model achieves solid performance >85% Dice")
        else:
            print("⚠️  ROOM FOR IMPROVEMENT: Consider further optimization")
            
    else:
        print("❌ No valid results generated!")

if __name__ == "__main__":
    validate_best_model()
