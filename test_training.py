import torch
import torch.nn as nn
from config.data_config import get_data_loaders
from module.model.proposed_net import DermoMamba
from loss.proposed_loss import Guide_Fusion_Loss
import time

def test_training_step():
    print("Setting up training test...")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    print(f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DermoMamba().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test one training step
    model.train()
    print("Testing one training step...")
    
    start_time = time.time()
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        # Squeeze the channel dimension if it's 1
        if outputs.shape[1] == 1:
            outputs_for_loss = outputs.squeeze(1)
        else:
            outputs_for_loss = outputs
        loss = Guide_Fusion_Loss(outputs_for_loss, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        print(f"Input shape: {images.shape}, Output shape: {outputs.shape}, Target shape: {masks.shape}")
        
        # Just test one batch
        break
    
    end_time = time.time()
    print(f"Training step completed in {end_time - start_time:.2f} seconds")
    
    # Test validation step
    model.eval()
    print("Testing one validation step...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            # Squeeze the channel dimension if it's 1
            if outputs.shape[1] == 1:
                outputs_for_loss = outputs.squeeze(1)
            else:
                outputs_for_loss = outputs
            loss = Guide_Fusion_Loss(outputs_for_loss, masks)
            
            print(f"Val Batch {batch_idx}: Loss = {loss.item():.4f}")
            break
    
    print("Training test completed successfully!")

if __name__ == "__main__":
    test_training_step()
