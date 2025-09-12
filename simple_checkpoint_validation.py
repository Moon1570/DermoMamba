"""
Simple validation script that works with the existing dataset structure
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
from module.model.optimized_dermomamba import OptimizedDermoMamba
from module.model.optimized_dermomamba_complete import OptimizedDermoMamba as CompleteOptimizedDermoMamba
from module.model.spatial_aware_dermomamba import SpatialAwareDermoMamba
from module.model.fast_dermomamba import FastDermoMamba
from module.model.tiny_dermomamba_resmamba import TinyDermoMambaWithResMamba
from module.model.proposed_net import DermoMamba
from utils.comprehensive_spatial_preprocessing import ComprehensiveSpatialEdgeTransform
from metric.metrics import dice_score, iou_score

class SimpleValidationDataset(Dataset):
    """Simple dataset that finds images and masks automatically"""
    
    def __init__(self, data_root, transform=None, limit_samples=None):
        self.data_root = data_root
        self.transform = transform
        
        # Find image directory
        image_dirs = ['train_images', 'val_images', 'images']
        self.image_dir = None
        for img_dir in image_dirs:
            test_path = os.path.join(data_root, img_dir)
            if os.path.exists(test_path):
                self.image_dir = test_path
                break
        
        if self.image_dir is None:
            raise ValueError(f"No image directory found in {data_root}")
        
        # Find mask directory
        mask_dirs = ['train_masks', 'val_masks', 'masks']
        self.mask_dir = None
        for mask_dir in mask_dirs:
            test_path = os.path.join(data_root, mask_dir)
            if os.path.exists(test_path):
                self.mask_dir = test_path
                break
        
        if self.mask_dir is None:
            raise ValueError(f"No mask directory found in {data_root}")
        
        print(f"Using images from: {self.image_dir}")
        print(f"Using masks from: {self.mask_dir}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        
        self.image_files = sorted(self.image_files)
        
        # Limit samples if specified
        if limit_samples:
            self.image_files = self.image_files[:limit_samples]
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Try different mask naming conventions
        possible_masks = [
            os.path.join(self.mask_dir, f"{img_name}_segmentation.png"),
            os.path.join(self.mask_dir, f"{img_name}_mask.png"),
            os.path.join(self.mask_dir, f"{img_name}.png"),
            os.path.join(self.mask_dir, f"{img_name}_segmentation.jpg"),
            os.path.join(self.mask_dir, f"{img_name}_mask.jpg"),
            os.path.join(self.mask_dir, f"{img_name}.jpg"),
        ]
        
        mask_path = None
        for mask_p in possible_masks:
            if os.path.exists(mask_p):
                mask_path = mask_p
                break
        
        if mask_path is None:
            raise ValueError(f"No mask found for {img_name}")
        
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
            if hasattr(self.transform, '__call__') and not hasattr(self.transform, 'transforms'):
                # Comprehensive spatial transform
                transformed = self.transform(image)
                return transformed, torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            else:
                # Albumentations transform
                transformed = self.transform(image=image, mask=mask)
                return transformed['image'], transformed['mask']
        else:
            # Default normalization
            image = image / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask

class SimpleCheckpointValidator:
    def __init__(self, data_path, image_size=224, batch_size=4, limit_samples=50):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Data path: {data_path}")
        
        # Setup validation data
        self.setup_validation_data(limit_samples)
        
    def setup_validation_data(self, limit_samples):
        """Setup validation dataset"""
        
        # Standard validation transforms
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Comprehensive spatial transform for 6-channel models
        self.spatial_transform = ComprehensiveSpatialEdgeTransform(
            input_size=(self.image_size, self.image_size),
            apply_all_methods=True,
            add_spatial_coords=True,
            edge_strength=0.7
        )
        
        # Create datasets
        try:
            # 3-channel dataset
            self.val_dataset_3ch = SimpleValidationDataset(
                data_root=self.data_path,
                transform=self.val_transform,
                limit_samples=limit_samples
            )
            
            # 6-channel dataset  
            self.val_dataset_6ch = SimpleValidationDataset(
                data_root=self.data_path,
                transform=self.spatial_transform,
                limit_samples=limit_samples
            )
            
            # Create dataloaders
            self.val_loader_3ch = DataLoader(
                self.val_dataset_3ch, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True
            )
            
            self.val_loader_6ch = DataLoader(
                self.val_dataset_6ch, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True
            )
            
            print(f"Created validation datasets:")
            print(f"  3-channel: {len(self.val_dataset_3ch)} samples")
            print(f"  6-channel: {len(self.val_dataset_6ch)} samples")
            
        except Exception as e:
            print(f"Error creating validation datasets: {e}")
            raise
    
    def load_model_for_checkpoint(self, checkpoint_path):
        """Load appropriate model architecture based on checkpoint analysis"""
        print(f"Analyzing checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint to inspect
            if checkpoint_path.endswith('.pth'):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint
            else:  # .ckpt files (Lightning)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            
            # Analyze state_dict to determine model type and input channels
            first_layer_key = None
            input_channels = 3  # default
            
            # Find the first convolutional layer to determine input channels
            for key in state_dict.keys():
                if 'pw_in.weight' in key:
                    first_layer_key = key
                    break
                elif 'patch_embed' in key and 'weight' in key:
                    first_layer_key = key
                    break
                elif 'stem' in key and 'weight' in key and 'conv' in key:
                    first_layer_key = key
                    break
                elif 'conv1.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key and first_layer_key in state_dict:
                input_channels = state_dict[first_layer_key].shape[1]
                print(f"Detected input channels: {input_channels}")
            
            # Determine model architecture
            model_type = "unknown"
            if 'spatial' in checkpoint_path.lower() or input_channels == 6:
                model_type = "spatial_aware"
                model = SpatialAwareDermoMamba(n_class=1, input_channels=input_channels)
            elif 'optimized' in checkpoint_path.lower() and 'complete' in checkpoint_path.lower():
                model_type = "optimized_complete"
                model = CompleteOptimizedDermoMamba(n_class=1)
            elif 'optimized' in checkpoint_path.lower():
                model_type = "optimized"
                model = OptimizedDermoMamba(n_class=1, use_gradient_checkpointing=False)
            elif 'fast' in checkpoint_path.lower():
                model_type = "fast"
                model = FastDermoMamba(n_class=1)
            elif 'tiny' in checkpoint_path.lower():
                model_type = "tiny"
                model = TinyDermoMambaWithResMamba(n_class=1)
            else:
                model_type = "standard"
                model = DermoMamba()
            
            print(f"Using {model_type} model with {input_channels} input channels")
            
            # Load weights
            if checkpoint_path.endswith('.ckpt'):
                # Lightning checkpoint - remove 'model.' prefix if present
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            
            model = model.to(self.device)
            model.eval()
            
            return model, input_channels, model_type
            
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
            return None, None, None
    
    def validate_checkpoint(self, checkpoint_path):
        """Validate a single checkpoint"""
        print(f"\n{'='*60}")
        print(f"Validating: {os.path.basename(checkpoint_path)}")
        print(f"{'='*60}")
        
        # Load model
        model, input_channels, model_type = self.load_model_for_checkpoint(checkpoint_path)
        if model is None:
            return None
        
        # Choose appropriate dataloader
        val_loader = self.val_loader_6ch if input_channels == 6 else self.val_loader_3ch
        print(f"Using {'6-channel' if input_channels == 6 else '3-channel'} validation data")
        
        # Initialize metrics tracking
        all_dice_scores = []
        all_iou_scores = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if batch_idx % 5 == 0:
                    print(f"  Processing batch {batch_idx}/{len(val_loader)}")

                images = images.to(self.device)
                masks = masks.to(self.device)

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
        
        if all_dice_scores:
            mean_dice = np.mean(all_dice_scores)
            std_dice = np.std(all_dice_scores)
            mean_iou = np.mean(all_iou_scores)
            std_iou = np.std(all_iou_scores)
            
            results = {
                'checkpoint_path': checkpoint_path,
                'model_type': model_type,
                'input_channels': input_channels,
                'dice_score': {
                    'mean': float(mean_dice),
                    'std': float(std_dice),
                    'percentage': float(mean_dice * 100)
                },
                'iou_score': {
                    'mean': float(mean_iou),
                    'std': float(std_iou), 
                    'percentage': float(mean_iou * 100)
                },
                'num_samples': len(all_dice_scores)
            }
            
            print(f"Results for {os.path.basename(checkpoint_path)}:")
            print(f"  Model Type: {model_type}")
            print(f"  Input Channels: {input_channels}")
            print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f} ({mean_dice*100:.2f}%)")
            print(f"  IoU Score:  {mean_iou:.4f} ± {std_iou:.4f} ({mean_iou*100:.2f}%)")
            print(f"  Samples: {len(all_dice_scores)}")
            
            return results
        else:
            print("No valid predictions generated!")
            return None
    
    def validate_all_checkpoints(self):
        """Validate all checkpoints in the checkpoints directory"""
        checkpoint_dir = Path("d:/Research/DermoMamba/checkpoints")
        results = []
        
        print(f"Searching for checkpoints in: {checkpoint_dir}")
        
        # Find all checkpoint files
        checkpoint_patterns = [
            "**/*.ckpt",
            "**/*.pth"
        ]
        
        checkpoint_files = []
        for pattern in checkpoint_patterns:
            checkpoint_files.extend(glob.glob(str(checkpoint_dir / pattern), recursive=True))
        
        checkpoint_files = sorted(checkpoint_files)
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Validate each checkpoint
        for checkpoint_path in checkpoint_files:
            try:
                result = self.validate_checkpoint(checkpoint_path)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Failed to validate {checkpoint_path}: {e}")
                continue
        
        return results
    
    def save_results(self, results):
        """Save validation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"d:/Research/DermoMamba/checkpoint_validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return results_file
    
    def print_summary(self, results):
        """Print a summary of all validation results"""
        if not results:
            print("No successful validations!")
            return
        
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        # Sort by Dice score
        results_sorted = sorted(results, key=lambda x: x['dice_score']['mean'], reverse=True)
        
        print(f"{'Rank':<4} {'Checkpoint':<30} {'Model':<15} {'Ch':<3} {'Dice %':<8} {'IoU %':<8}")
        print("-" * 70)
        
        for i, result in enumerate(results_sorted, 1):
            checkpoint_name = os.path.basename(result['checkpoint_path'])[:29]
            model_type = result['model_type'][:14]  # Truncate if too long
            channels = result['input_channels']
            dice_pct = result['dice_score']['percentage']
            iou_pct = result['iou_score']['percentage']
            
            print(f"{i:<4} {checkpoint_name:<30} {model_type:<15} {channels:<3} {dice_pct:<8.2f} {iou_pct:<8.2f}")
        
        # Find best model
        best_result = results_sorted[0]
        print(f"\n🏆 BEST MODEL:")
        print(f"   Checkpoint: {os.path.basename(best_result['checkpoint_path'])}")
        print(f"   Model Type: {best_result['model_type']}")
        print(f"   Dice Score: {best_result['dice_score']['percentage']:.2f}%")
        print(f"   IoU Score:  {best_result['iou_score']['percentage']:.2f}%")
        
        # Check if we found the 89.25% model
        high_performing = [r for r in results if r['dice_score']['percentage'] > 89.0]
        if high_performing:
            print(f"\n🎯 HIGH PERFORMING MODELS (>89%):")
            for result in high_performing:
                print(f"   {os.path.basename(result['checkpoint_path'])}: {result['dice_score']['percentage']:.2f}% Dice")

def main():
    print("Starting simple checkpoint validation...")
    
    # Set data path
    data_path = "d:/Research/DermoMamba/data/ISIC2018_test"
    if not os.path.exists(data_path):
        # Try alternative paths
        alt_paths = [
            "d:/Research/DermoMamba/data/ISIC2018",
            "d:/Research/DermoMamba/data/ISIC2018_proc"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
    
    print(f"Using data path: {data_path}")
    
    # Initialize validator with limited samples for faster testing
    validator = SimpleCheckpointValidator(data_path=data_path, limit_samples=50)
    
    # Run validation
    results = validator.validate_all_checkpoints()
    
    # Save and print results
    if results:
        validator.save_results(results)
        validator.print_summary(results)
        
        print(f"\n✅ Validation complete! Tested {len(results)} models successfully.")
    else:
        print("\n❌ No successful validations completed!")

if __name__ == "__main__":
    main()
