"""
Test script to verify the DermoMamba implementation
"""
import torch
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model():
    """Test model forward pass"""
    print("Testing DermoMamba model...")
    
    from module.model.proposed_net import DermoMamba
    
    # Create model
    model = DermoMamba()
    print(f"Model created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 192, 256)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min/max: {output.min():.4f}/{output.max():.4f}")
    
    return True

def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        from config.data_config import train_dataset, test_dataset
        
        print(f"Train dataset size: {len(train_dataset.dataset)}")
        print(f"Validation dataset size: {len(test_dataset.dataset)}")
        
        # Test loading one batch
        for batch_idx, (images, masks) in enumerate(train_dataset):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Masks shape: {masks.shape}")
            print(f"  Images min/max: {images.min():.3f}/{images.max():.3f}")
            print(f"  Masks min/max: {masks.min():.3f}/{masks.max():.3f}")
            break
        
        return True
    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False

def test_loss():
    """Test Guide Fusion Loss"""
    print("\nTesting Guide Fusion Loss...")
    
    try:
        from loss.proposed_loss import Guide_Fusion_Loss
        
        # Create dummy data
        batch_size = 2
        pred = torch.randn(batch_size, 1, 192, 256)
        target = torch.randint(0, 2, (batch_size, 1, 192, 256)).float()
        
        # Test loss computation
        loss = Guide_Fusion_Loss(pred, target)
        print(f"Loss computed successfully: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"Loss test failed: {e}")
        return False

def test_segmentor():
    """Test segmentor module"""
    print("\nTesting Segmentor...")
    
    try:
        from config.segmentor import Segmentor
        from module.model.proposed_net import DermoMamba
        
        model = DermoMamba()
        segmentor = Segmentor(model=model)
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 192, 256)
        target = torch.randint(0, 2, (batch_size, 1, 192, 256)).float()
        
        segmentor.eval()
        with torch.no_grad():
            output = segmentor(input_tensor)
        
        print(f"Segmentor forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"Segmentor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running DermoMamba implementation tests...\n")
    
    tests = [
        test_model,
        test_loss,
        test_segmentor,
        test_dataset
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with error: {e}")
            results.append(False)
    
    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"{'='*50}")
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_func.__name__}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()
