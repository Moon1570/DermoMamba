"""
Training script for DermoMamb    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    # Create datasets first (instead of directly getting dataloaders)
    from torch.utils.data import DataLoader
    from datasets.isic_dataset import ISICDataset, get_train_transforms, get_val_transforms
    
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
    
    # Create manual dataloaders with generator=torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator()  # Use CPU-based generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator()  # Use CPU-based generator
    )IC 2018 dataset
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config.segmentor import Segmentor
from module.model.proposed_net import DermoMamba
from datasets.isic_dataset import create_data_loaders

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n{'='*50}")
        print(f"🔥 GPU DETECTED: {device_name}")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
        print(f"💾 Total GPU Memory: {memory_total:.2f} GB")
        print(f"💾 Initial Memory Usage: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")
        print(f"{'='*50}\n")
        
        # Don't set default tensor type to avoid issues with dataloaders
    else:
        print("\n❌ WARNING: No GPU detected, running on CPU which will be very slow!\n")
    
    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_root=data_root,
        train_split=train_split,
        val_split=val_split,
        batch_size=16,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = DermoMamba()
    
    # Explicitly move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"👉 Model initialized and moved to: {device}")
    
    segmentor = Segmentor(model=model)
    
    # Create output directory
    os.makedirs('./checkpoints/ISIC2018/', exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/ISIC2018/',
        filename='dermomamba-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        verbose=True,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
    
    early_stopping = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar()
    
    # Logger
    logger = TensorBoardLogger("tb_logs", name="dermomamba")
    
    # Trainer configuration
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        print(f"🚀 Training will use GPU acceleration with {devices} device(s)")
    else:
        accelerator = 'cpu'
        devices = 1
        print(f"⚠️ Training will use CPU (much slower)")
        
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stopping, progress_bar],
        logger=logger,
        precision='16-mixed',
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        num_sanity_val_steps=0,  # Skip sanity check
        enable_progress_bar=True,
        benchmark=True
    )
    
    # Start training
    print("\nStarting training process...")
    
    # Check if model is on GPU
    is_on_gpu = next(segmentor.parameters()).is_cuda
    print(f"🔍 Model running on GPU: {'✅ YES' if is_on_gpu else '❌ NO'}")
    
    if is_on_gpu:
        print(f"📊 Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB allocated")
        print(f"⚡ Beginning training on GPU: {torch.cuda.get_device_name(0)}\n")
    
    trainer.fit(segmentor, train_loader, val_loader)
    
    # Test on validation set
    trainer.test(segmentor, val_loader, ckpt_path='best')

if __name__ == "__main__":
    main()
