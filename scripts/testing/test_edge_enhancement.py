#!/usr/bin/env python3
"""
Test Edge Enhancement Effects
This script visualizes how different edge enhancement methods affect skin lesion images
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.edge_preprocessing import (
    EdgeEnhancedTransform, 
    GradientEnhancedTransform,
    get_edge_enhanced_train_transforms,
    get_boundary_optimized_transforms
)

def test_edge_enhancement():
    print("🔍 Testing Edge Enhancement Effects")
    print("="*50)
    
    # Find a sample image
    sample_image_path = None
    test_paths = [
        "data/ISIC2018/train_images",
        "data/ISIC2018_proc/train_images", 
        "data/ISIC2018_test/train_images"
    ]
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            images = [f for f in os.listdir(test_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                sample_image_path = os.path.join(test_path, images[0])
                break
    
    if not sample_image_path:
        print("❌ No sample images found. Please ensure data is available.")
        return
    
    print(f"📷 Using sample image: {sample_image_path}")
    
    # Load and convert image
    img = np.array(Image.open(sample_image_path).convert('RGB'))
    
    # Test different enhancement methods
    methods = {
        'Original': None,
        'Unsharp Mask': EdgeEnhancedTransform(enhance_method='unsharp_mask', strength=0.4, p=1.0),
        'Laplacian': EdgeEnhancedTransform(enhance_method='laplacian', strength=0.4, p=1.0),
        'Sobel': EdgeEnhancedTransform(enhance_method='sobel', strength=0.4, p=1.0),
        'CLAHE + Edges': EdgeEnhancedTransform(enhance_method='clahe_edges', strength=0.4, p=1.0),
        'Gradient Enhanced': GradientEnhancedTransform(gradient_strength=0.3, p=1.0)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (method_name, transform) in enumerate(methods.items()):
        if transform is None:
            processed_img = img
        else:
            processed_img = transform.apply(img)
        
        axes[i].imshow(processed_img)
        axes[i].set_title(f'{method_name}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
        # Add enhancement info
        if transform is not None:
            axes[i].text(0.02, 0.02, '✓ Enhanced', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                        fontsize=10, color='white', weight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    save_path = "edge_enhancement_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Comparison saved to: {save_path}")
    plt.show()
    
    # Test the albumentations transforms
    print("\n🔄 Testing Albumentations Integration...")
    
    try:
        # Test boundary optimized transforms
        boundary_transform = get_boundary_optimized_transforms()
        enhanced_img = boundary_transform(image=img)['image']
        print(f"✅ Boundary optimized transform: {enhanced_img.shape}")
        
        # Test edge enhanced transforms  
        edge_transform = get_edge_enhanced_train_transforms(edge_enhancement='unsharp_mask')
        enhanced_img2 = edge_transform(image=img)['image']
        print(f"✅ Edge enhanced transform: {enhanced_img2.shape}")
        
        print("✅ All transforms working correctly!")
        
    except Exception as e:
        print(f"❌ Transform error: {e}")
    
    print("\n📊 Enhancement Summary:")
    print("• Unsharp Mask: Best for general edge sharpening")
    print("• Laplacian: Good for fine edge details") 
    print("• Sobel: Strong gradient-based enhancement")
    print("• CLAHE + Edges: Best for low-contrast lesions")
    print("• Gradient Enhanced: Adaptive enhancement based on gradients")
    
    return save_path

if __name__ == "__main__":
    test_edge_enhancement()
