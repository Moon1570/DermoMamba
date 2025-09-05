"""
Optimized training script for DermoMamba with performance improvements
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from config.segmentor import Segmentor
from module.model.optimized_dermomamba import OptimizedDermoMamba
from datasets.isic_dataset import ISICDataset, get_train_transforms, get_val_transforms

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n{'='*50}")
        print(f"🔥 GPU DETECTED: {device_name}")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
        print(f"💾 Total GPU Memory: {memory_total:.2f} GB")
        print(f"{'='*50}\n")
    else:
        print("\n❌ WARNING: No GPU detected, running on CPU which will be very slow!\n")

    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    # Create datasets with smaller resolution for faster training
    train_dataset = ISICDataset(
        data_root=data_root,
        split_file=train_split,
        transform=get_train_transforms(input_size=(96, 128)),  # Reduced size
        is_train=True
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split_file=val_split,
        transform=get_val_transforms(input_size=(96, 128)),   # Reduced size
        is_train=False
    )
    
    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Reduced batch size
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid memory issues
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=True,
        persistent_workers=False
    )
    
    print(f"Training samples: {len(train_dataset)} (batch size: 2)")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    # Initialize optimized model
    model = OptimizedDermoMamba(use_gradient_checkpointing=True)
    print("✅ Using OptimizedDermoMamba with gradient checkpointing enabled")
    
    # Move model to GPU explicitly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"👉 Model initialized and moved to: {device}")
    
    segmentor = Segmentor(model=model)
    
    # Create output directory
    os.makedirs('./checkpoints/ISIC2018/', exist_ok=True)
    
    # Callbacks with less frequent checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/ISIC2018/',
        filename='dermomamba-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=1,  # Only save best model
        verbose=True,
        save_weights_only=True,  # Save only weights to save space
        auto_insert_metric_name=False,
        every_n_epochs=5  # Save every 5 epochs
    )
    
    early_stopping = EarlyStopping(
        monitor='val_dice',
        patience=10,  # Reduced patience
        mode='max',
        verbose=True
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)  # Less frequent updates
    
    # Logger
    logger = TensorBoardLogger("tb_logs", name="dermomamba")
    
    # Trainer configuration with optimizations
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        print(f"🚀 Training will use GPU acceleration")
    else:
        accelerator = 'cpu'
        devices = 1
        print(f"⚠️ Training will use CPU")
        
    trainer = pl.Trainer(
        max_epochs=50,  # Reduced epochs for testing
        callbacks=[checkpoint_callback, early_stopping, progress_bar],
        logger=logger,
        precision='16-mixed',
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=20,  # Less frequent logging
        val_check_interval=0.5,  # Validate twice per epoch
        num_sanity_val_steps=1,  # Minimal sanity check
        enable_progress_bar=True,
        benchmark=True,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=2,  # Simulate larger batch size
        enable_checkpointing=True,
        enable_model_summary=False,  # Disable model summary to save time
        detect_anomaly=False,  # Disable anomaly detection for speed
    )
    
    # Start training
    print("\nStarting optimized training process...")
    
    # Check if model is on GPU
    is_on_gpu = next(segmentor.parameters()).is_cuda
    print(f"🔍 Model running on GPU: {'✅ YES' if is_on_gpu else '❌ NO'}")
    
    if is_on_gpu:
        print(f"📊 Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB allocated")
        print(f"⚡ Beginning training on GPU: {torch.cuda.get_device_name(0)}\n")
    
    try:
        trainer.fit(segmentor, train_loader, val_loader)
        print("✅ Training completed successfully!")
        
        # Test on validation set
        trainer.test(segmentor, val_loader, ckpt_path='best')
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared")

if __name__ == "__main__":
    main()
