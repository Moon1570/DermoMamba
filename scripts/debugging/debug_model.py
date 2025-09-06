"""
Debug script to diagnose issues with DermoMamba model training
"""
import torch
import sys
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.model.proposed_net import DermoMamba
from datasets.isic_dataset import ISICDataset, get_train_transforms, get_val_transforms
from config.segmentor import Segmentor

def check_basic_model_forward():
    """Basic test of model forward pass"""
    print("\n===== BASIC MODEL FORWARD PASS TEST =====")
    
    # Create model
    model = DermoMamba()
    print(f"Model created successfully!")

    # Test forward pass with verbose error handling
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 192, 256)

    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

def check_gpu_model_forward():
    """Test model forward pass on GPU"""
    print("\n===== GPU MODEL FORWARD PASS TEST =====")
    
    if not torch.cuda.is_available():
        print("No GPU available. Skipping GPU test.")
        return
        
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Create model and move to GPU
    model = DermoMamba().to('cuda')
    print(f"Model created and moved to device: {next(model.parameters()).device}")
    print(f"GPU memory after model creation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Test forward pass on GPU
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 192, 256, device='cuda')
    print(f"Input tensor created on device: {input_tensor.device}")
    print(f"GPU memory after input creation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    try:
        model.eval()
        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = model(input_tensor)
            end_event.record()
            
            # Wait for GPU to finish work
            torch.cuda.synchronize()
            
            # Calculate time
            elapsed_time = start_event.elapsed_time(end_event)
            
        print(f"Success! Output shape: {output.shape}, device: {output.device}")
        print(f"Forward pass took {elapsed_time:.2f} ms")
        print(f"GPU memory after forward pass: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error occurred on GPU: {e}")
        import traceback
        traceback.print_exc()

def check_data_loading():
    """Test data loading from dataset"""
    print("\n===== DATA LOADING TEST =====")
    
    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    
    try:
        # Create dataset
        train_dataset = ISICDataset(
            data_root=data_root,
            split_file=train_split,
            transform=get_train_transforms(),
            is_train=True
        )
        print(f"Dataset created successfully with {len(train_dataset)} samples")
        
        # Create dataloader with 0 workers (safer for debugging)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,  # Don't shuffle for deterministic results
            num_workers=0,  # No multiprocessing to simplify debugging
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"DataLoader created with {len(train_loader)} batches")
        
        # Test loading a batch
        print("Loading first batch...")
        first_batch = next(iter(train_loader))
        images, masks = first_batch
        
        print(f"First batch loaded successfully!")
        print(f"Images shape: {images.shape}, device: {images.device}")
        print(f"Masks shape: {masks.shape}, device: {masks.device}")
        
        if torch.cuda.is_available():
            # Try moving to GPU
            print("Moving batch to GPU...")
            images = images.cuda()
            masks = masks.cuda()
            print(f"Images device after move: {images.device}")
            print(f"Masks device after move: {masks.device}")
    
    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()

def check_pl_trainer():
    """Test PyTorch Lightning trainer setup"""
    print("\n===== PYTORCH LIGHTNING TRAINER TEST =====")
    
    # Data paths
    data_root = "D:/Research/DermoMamba/data/ISIC2018_proc"
    train_split = "splits/isic2018_train.txt"
    val_split = "splits/isic2018_val.txt"
    
    try:
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
        
        # Use 0 workers for testing
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize model
        print("Creating model and segmentor...")
        model = DermoMamba()
        
        # Check device before segmentor creation
        print(f"Model device before segmentor: {next(model.parameters()).device}")
        
        segmentor = Segmentor(model=model)
        print(f"Segmentor device after creation: {next(segmentor.parameters()).device}")
        
        # Setup minimal trainer
        print("Creating trainer...")
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,  # Only use 2 batches
            limit_val_batches=1,    # Only use 1 batch
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_checkpointing=False,  # Disable checkpoints
            logger=False,  # Disable logging
            enable_progress_bar=True,
            num_sanity_val_steps=0,  # Skip sanity check
        )
        
        print(f"Trainer created with accelerator: {trainer.accelerator}")
        print("Training for 1 epoch with limited batches...")
        
        # Run fit
        trainer.fit(segmentor, train_loader, val_loader)
        
        print("Training completed successfully!")
        print(f"Final model device: {next(segmentor.parameters()).device}")
        
    except Exception as e:
        print(f"Error in PyTorch Lightning training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("===== DERMOMAMBA DEBUG SCRIPT =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run tests
    check_basic_model_forward()
    check_gpu_model_forward()
    check_data_loading()
    check_pl_trainer()
