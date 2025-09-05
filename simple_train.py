"""
Simple training script for DermoMamba without PyTorch Lightning
This will help diagnose issues with the GPU training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
import os

from module.model.proposed_net import DermoMamba
from datasets.isic_dataset import ISICDataset, get_train_transforms, get_val_transforms
# Use basic BCE loss instead of the custom loss
import torch.nn.functional as F
# Simple dice metric
def simple_dice_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()
    
    intersection = (y_pred * y_true).sum()
    return (2. * intersection) / (y_pred.sum() + y_true.sum() + 1e-7)

def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Create model and move to device
    model = DermoMamba().to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
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
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 1  # Just for testing
    
    print("\nStarting training loop...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        batch_count = 0
        
        # Simple progress indicator
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_batches = min(10, len(train_loader))  # Process at most 10 batches for testing
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            if batch_idx >= 10:  # Just process a few batches for testing
                break
            
            # Print simple progress indicator
            progress = (batch_idx + 1) / total_batches * 100
            print(f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%)... ", end="", flush=True)
                
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Ensure masks have the right shape (add channel dimension if needed)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass
            start_time = time.time()
            outputs = model(images)
            forward_time = time.time() - start_time
            
            # Calculate simple BCE loss
            loss = F.binary_cross_entropy_with_logits(outputs, masks.float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = simple_dice_score(outputs, masks)
            
            # Update statistics
            train_loss += loss.item()
            train_dice += dice
            batch_count += 1
            
            # Print batch results
            print(f"loss={loss.item():.4f}, dice={dice:.4f}, fw_time={forward_time*1000:.1f}ms")
            
            # Print memory usage
            if device.type == 'cuda':
                print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB allocated, "
                      f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB reserved")
        
        # Calculate average metrics
        avg_loss = train_loss / batch_count
        avg_dice = train_dice / batch_count
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
