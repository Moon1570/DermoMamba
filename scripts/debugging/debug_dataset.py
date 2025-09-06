"""
Debug dataset loading issue
"""
import torch
import sys
import os
sys.path.append('.')

def test_dataset_loading():
    print("Testing dataset loading...")
    
    from datasets.isic_dataset import ISICDataset
    from torch.utils.data import DataLoader
    
    # Dataset paths
    data_root = "d:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "d:/Research/DermoMamba/splits/isic2018_train.txt"
    
    try:
        # Create dataset
        dataset = ISICDataset(
            data_root=data_root,
            split_file=train_split,
            is_train=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Try to load one sample
        print("Loading first sample...")
        image, mask = dataset[0]
        
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image type: {type(image)}")
        print(f"Mask type: {type(mask)}")
        
        # Try data loader
        print("Creating data loader...")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        print("Loading first batch...")
        batch = next(iter(loader))
        images, masks = batch
        
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
        print("✅ Dataset loading successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading()
