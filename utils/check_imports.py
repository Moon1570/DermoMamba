"""
Quick check of the complete optimized model imports
"""
import sys
sys.path.append('.')

try:
    print("Testing imports...")
    from module.model.optimized_dermomamba_complete import OptimizedDermoMamba
    print("✅ Model import successful")
    
    from datasets.isic_dataset import ISICDataset
    print("✅ Dataset import successful")
    
    from loss.loss import DiceLoss, calc_loss, bce_tversky_loss
    print("✅ Loss import successful")
    
    from metric.metrics import dice_score, iou_score
    print("✅ Metrics import successful")
    
    import torch
    import pytorch_lightning as pl
    print("✅ PyTorch Lightning import successful")
    
    # Test model creation
    model = OptimizedDermoMamba(n_class=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully: {total_params:,} parameters")
    
    print("✅ All imports and model creation successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
