"""
Training script for TinyDermoMamba with ResMambaBlock
Uses proper DermoMamba architecture with memory optimizations
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Set memory management
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
        
        return device
    else:
        print("❌ GPU not available, using CPU")
        return torch.device('cpu')

def main():
    print("="*60)
    print("TinyDermoMamba with ResMambaBlock Training")
    print("="*60)
    
    # Setup
    device = setup_gpu()
    
    # Import after path setup
    from datasets.isic_dataset import ISICDataset
    from module.model.tiny_dermomamba_resmamba import TinyDermoMambaWithResMamba
    from loss.loss import DiceLoss, calc_loss
    from metric.metrics import dice_score, iou_score
    
    # Dataset paths
    data_root = "d:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "d:/Research/DermoMamba/splits/isic2018_train.txt"
    val_split = "d:/Research/DermoMamba/splits/isic2018_val.txt"
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"❌ Training data not found at {data_root}")
        return
    
    print(f"✅ Training data found at {data_root}")
    
    # Create datasets
    train_dataset = ISICDataset(
        data_root=data_root,
        split_file=train_split,
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split_file=val_split,
        is_train=False
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create data loaders with small batch size due to slower model
    batch_size = 1 if device.type == 'cuda' else 1  # Very small batch due to slow inference
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Reduced workers
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Create model
    model = TinyDermoMambaWithResMamba(n_class=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Loss and optimizer
    def combined_loss(pred, target):
        return calc_loss(pred, target, bce_weight=0.5)
    
    criterion = combined_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)  # Shorter schedule
    
    # Mixed precision training
    scaler = GradScaler()
    
    print("✅ Setup complete, starting training...")
    print("="*60)
    
    # Test one forward pass first
    print("Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_images, test_masks = test_batch
        test_images = test_images.to(device)
        test_masks = test_masks.to(device)
        
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
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"✅ GPU Memory: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")
    
    print("="*60)
    
    # Training loop - reduced epochs due to slower training
    num_epochs = 25
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                # Resize outputs to match mask size if needed
                if outputs.size()[2:] != masks.size()[2:]:
                    outputs = F.interpolate(outputs, size=masks.size()[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            train_loss += loss.item()
            with torch.no_grad():
                dice = dice_score(outputs, masks)
                train_dice += dice.item()
            
            # Clear cache periodically
            if batch_idx % 25 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Print progress every 200 batches (less frequent due to slow training)
            if (batch_idx + 1) % 200 == 0:
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed
                current_memory = torch.cuda.memory_allocated() / 1024**2 if device.type == 'cuda' else 0
                eta_minutes = ((len(train_loader) - batch_idx - 1) / batches_per_sec) / 60 if batches_per_sec > 0 else 0
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Dice: {dice.item():.4f}, "
                      f"Speed: {batches_per_sec:.3f} b/s, "
                      f"Memory: {current_memory:.1f}MB, "
                      f"ETA: {eta_minutes:.1f}min")
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Validation phase - sample only 50 batches for speed
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_batches = min(50, len(val_loader))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= val_batches:
                    break
                    
                images, masks = batch
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    if outputs.size()[2:] != masks.size()[2:]:
                        outputs = F.interpolate(outputs, size=masks.size()[2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                dice = dice_score(outputs, masks)
                val_dice += dice.item()
        
        val_loss /= val_batches
        val_dice /= val_batches
        
        # Update learning rate
        scheduler.step()
        
        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Print epoch summary
        current_memory = torch.cuda.memory_allocated() / 1024**2 if device.type == 'cuda' else 0
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} (on {val_batches} batches)")
        print(f"  Time: {epoch_time/60:.1f}min, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Memory: {current_memory:.1f}MB")
        
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
            os.makedirs('checkpoints/resmamba', exist_ok=True)
            torch.save(checkpoint, 'checkpoints/resmamba/best_model.pth')
            print(f"  ✅ New best model saved! Dice: {best_dice:.4f}")
        
        print("-" * 60)
    
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
