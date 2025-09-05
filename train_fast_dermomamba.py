"""
Ultra-fast training script for DermoMamba
Uses the most efficient approximations for maximum speed
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# GPU Detection and setup
def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        return device
    else:
        print("❌ GPU not available, using CPU")
        return torch.device('cpu')

def main():
    print("="*60)
    print("Ultra-Fast DermoMamba Training")
    print("="*60)
    
    # Setup
    device = setup_gpu()
    
    # Import after path setup
    from datasets.isic_dataset import ISICDataset
    from module.model.fast_dermomamba import FastDermoMamba
    from loss.proposed_loss import ProposedLoss
    from metric.metrics import DiceScore
    
    # Dataset paths
    train_img_path = "d:/Research/DermoMamba/data/ISIC2018/train_images"
    train_mask_path = "d:/Research/DermoMamba/data/ISIC2018/train_masks"
    val_img_path = "d:/Research/DermoMamba/data/ISIC2018/val_images"
    val_mask_path = "d:/Research/DermoMamba/data/ISIC2018/val_images"  # Note: using val_images for both (as in original)
    
    # Check if data exists
    if not os.path.exists(train_img_path):
        print(f"❌ Training data not found at {train_img_path}")
        return
    
    print(f"✅ Training data found at {train_img_path}")
    
    # Create datasets with smaller image size for speed
    train_dataset = ISICDataset(
        img_path=train_img_path,
        mask_path=train_mask_path,
        image_size=224,  # Smaller size for speed
        mode='train'
    )
    
    val_dataset = ISICDataset(
        img_path=val_img_path,
        mask_path=val_mask_path,
        image_size=224,  # Smaller size for speed
        mode='val'
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create data loaders with larger batch size for speed
    batch_size = 8 if device.type == 'cuda' else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Create model
    model = FastDermoMamba(n_class=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = ProposedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Metrics
    dice_metric = DiceScore()
    
    print("✅ Setup complete, starting training...")
    print("="*60)
    
    # Test one forward pass first
    print("Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_images = test_batch['image'].to(device)
        test_masks = test_batch['mask'].to(device)
        
        start_time = time.time()
        with autocast():
            test_outputs = model(test_images)
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"✅ Input shape: {test_images.shape}")
        print(f"✅ Output shape: {test_outputs.shape}")
        print(f"✅ Forward pass time: {forward_time:.3f}s")
        
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"✅ GPU Memory allocated: {memory_allocated:.2f} MB")
    
    print("="*60)
    
    # Training loop
    num_epochs = 50
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            train_loss += loss.item()
            with torch.no_grad():
                dice = dice_metric(torch.sigmoid(outputs), masks)
                train_dice += dice.item()
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Dice: {dice.item():.4f}, "
                      f"Speed: {batches_per_sec:.2f} batches/sec")
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                dice = dice_metric(torch.sigmoid(outputs), masks)
                val_dice += dice.item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }
            os.makedirs('checkpoints/fast', exist_ok=True)
            torch.save(checkpoint, 'checkpoints/fast/best_model.pth')
            print(f"  ✅ New best model saved! Dice: {best_dice:.4f}")
        
        print("-" * 60)
    
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
