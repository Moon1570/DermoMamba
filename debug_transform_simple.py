#!/usr/bin/env python3
"""
Simple test of just the ComprehensiveSpatialEdgeTransform
"""

import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.comprehensive_spatial_preprocessing import ComprehensiveSpatialEdgeTransform

def test_raw_transform():
    """Test the raw transform without any additional processing"""
    print("🔬 TESTING RAW COMPREHENSIVE SPATIAL EDGE TRANSFORM")
    print("="*60)
    
    # Create transform
    transform = ComprehensiveSpatialEdgeTransform(input_size=(384, 384))  # Match input size
    print("✅ Created ComprehensiveSpatialEdgeTransform")
    
    # Load a sample image
    data_root = "data/ISIC2018_proc"
    train_images_dir = os.path.join(data_root, "train_images")
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        print("❌ No images found")
        return False
    
    # Load image
    sample_image_path = os.path.join(train_images_dir, image_files[0])
    original_image = Image.open(sample_image_path).convert('RGB')
    original_array = np.array(original_image)
    
    print(f"📷 Original image shape: {original_array.shape}")
    print(f"📷 Original image dtype: {original_array.dtype}")
    
    # Apply JUST the transform
    print("🔄 Applying ComprehensiveSpatialEdgeTransform...")
    transformed = transform(original_array)  # Use callable interface
    
    print(f"✅ Transform result shape: {transformed.shape}")
    print(f"✅ Transform result dtype: {transformed.dtype}")
    
    if transformed.shape[-1] == 6:
        print("🎉 SUCCESS: 6-channel output confirmed!")
        
        # Analyze channels
        print("\n📊 CHANNEL ANALYSIS:")
        channel_names = ["R", "G", "B", "X", "Y", "R_dist"]
        for i in range(6):
            ch_data = transformed[:,:,i]
            print(f"  Channel {i} ({channel_names[i]}): "
                  f"min={ch_data.min()}, max={ch_data.max()}, "
                  f"mean={ch_data.mean():.2f}")
        
        return True
    else:
        print(f"❌ Wrong channel count: {transformed.shape[-1]} (expected 6)")
        return False

if __name__ == "__main__":
    test_raw_transform()
