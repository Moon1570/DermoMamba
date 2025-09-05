import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, data_root, split_file, transform=None, is_train=True):
        """
        ISIC 2018 Dataset Loader
        
        Args:
            data_root: Path to data directory (e.g., "D:/Research/DermoMamba/data/ISIC2018_proc")
            split_file: Path to split file (e.g., "splits/isic2018_train.txt")
            transform: Data augmentation transforms
            is_train: Whether this is training dataset
        """
        self.data_root = os.path.normpath(data_root)
        self.is_train = is_train
        self.transform = transform
        
        # Read split file to get image names
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        # Always use train_images and train_masks for both train and validation
        self.image_dir = os.path.normpath(os.path.join(data_root, 'train_images'))
        self.mask_dir = os.path.normpath(os.path.join(data_root, 'train_masks'))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Get image name
        img_name = self.image_names[idx]
        
        # Load image and mask using normpath for consistent separators
        img_path = os.path.normpath(os.path.join(self.image_dir, f"{img_name}.png"))
        mask_path = os.path.normpath(os.path.join(self.mask_dir, f"{img_name}_segmentation.png"))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Normalize mask to 0-1
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default normalization
            image = image / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

def get_train_transforms(input_size=(192, 256)):
    """Training augmentations using Albumentations"""
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(input_size=(192, 256)):
    """Validation transforms"""
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_data_loaders(data_root, train_split, val_split, batch_size=4, num_workers=2):
    """Create train and validation data loaders"""
    
    # Create datasets
    train_dataset = ISICDataset(
        data_root=data_root,
        split_file=train_split,
        transform=get_train_transforms(),
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split_file=val_split,
        transform=get_val_transforms(),
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    train_loader, val_loader = create_data_loaders(
        data_root, train_split, val_split, batch_size=2
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Test loading one batch
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Image min/max: {images.min():.3f}/{images.max():.3f}")
        print(f"Mask min/max: {masks.min():.3f}/{masks.max():.3f}")
        break
