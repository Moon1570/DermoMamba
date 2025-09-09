"""
Enhanced Edge-Aware Preprocessing for Boundary Loss Optimization
This module provides advanced preprocessing techniques to enhance edge visibility
and improve boundary-aware loss function performance.
"""

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import torch.nn.functional as F

class EdgeEnhancedTransform(A.ImageOnlyTransform):
    """
    Custom transform to enhance edges in medical images for better boundary detection
    """
    
    def __init__(self, 
                 enhance_method='unsharp_mask',
                 strength=0.5,
                 always_apply=False,
                 p=1.0):
        super().__init__(always_apply, p)
        self.enhance_method = enhance_method
        self.strength = strength
    
    def apply(self, img, **params):
        if self.enhance_method == 'unsharp_mask':
            return self._unsharp_mask(img)
        elif self.enhance_method == 'laplacian':
            return self._laplacian_enhancement(img)
        elif self.enhance_method == 'sobel':
            return self._sobel_enhancement(img)
        elif self.enhance_method == 'clahe_edges':
            return self._clahe_edge_enhancement(img)
        else:
            return img
    
    def _unsharp_mask(self, img):
        """Apply unsharp masking to enhance edges"""
        # Convert to float32 for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(img_float, (9, 9), 2.0)
        
        # Create unsharp mask
        unsharp = img_float + self.strength * (img_float - blurred)
        
        # Clip and convert back
        unsharp = np.clip(unsharp * 255, 0, 255).astype(np.uint8)
        return unsharp
    
    def _laplacian_enhancement(self, img):
        """Enhance edges using Laplacian filter"""
        # Convert to grayscale for edge detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        
        # Normalize and enhance
        laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
        
        # Blend with original
        if len(img.shape) == 3:
            # Convert back to RGB
            laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            enhanced = cv2.addWeighted(img, 1.0 - self.strength, laplacian_rgb, self.strength, 0)
        else:
            enhanced = cv2.addWeighted(img, 1.0 - self.strength, laplacian, self.strength, 0)
            
        return enhanced
    
    def _sobel_enhancement(self, img):
        """Enhance edges using Sobel operators"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Apply Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = (sobel_combined / sobel_combined.max() * 255).astype(np.uint8)
        
        # Blend with original
        if len(img.shape) == 3:
            sobel_rgb = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB)
            enhanced = cv2.addWeighted(img, 1.0 - self.strength, sobel_rgb, self.strength, 0)
        else:
            enhanced = cv2.addWeighted(img, 1.0 - self.strength, sobel_combined, self.strength, 0)
            
        return enhanced
    
    def _clahe_edge_enhancement(self, img):
        """Combine CLAHE with edge enhancement for medical images"""
        # Apply CLAHE to each channel
        if len(img.shape) == 3:
            # Convert to LAB color space for better CLAHE results
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply to L channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
        
        # Add edge enhancement
        return self._unsharp_mask(enhanced)

class GradientEnhancedTransform(A.ImageOnlyTransform):
    """
    Transform that enhances gradients in the image to help boundary loss
    """
    
    def __init__(self, 
                 gradient_strength=0.3,
                 always_apply=False,
                 p=1.0):
        super().__init__(always_apply, p)
        self.gradient_strength = gradient_strength
    
    def apply(self, img, **params):
        # Convert to float32
        img_float = img.astype(np.float32) / 255.0
        
        # Calculate gradients
        if len(img.shape) == 3:
            # Process each channel
            enhanced = np.zeros_like(img_float)
            for c in range(img.shape[2]):
                enhanced[:,:,c] = self._enhance_gradients(img_float[:,:,c])
        else:
            enhanced = self._enhance_gradients(img_float)
        
        # Convert back to uint8
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return enhanced
    
    def _enhance_gradients(self, channel):
        """Enhance gradients in a single channel"""
        # Calculate gradients using Sobel
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        if grad_mag.max() > 0:
            grad_mag = grad_mag / grad_mag.max()
        
        # Enhance channel based on gradient magnitude
        enhanced = channel + self.gradient_strength * grad_mag
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced

def get_edge_enhanced_train_transforms(input_size=(384, 384), edge_enhancement='unsharp_mask'):
    """
    Training transforms with edge enhancement for boundary loss optimization
    """
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        
        # Standard augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        
        # Enhanced contrast and brightness for edge visibility
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.7),
        
        # Add some noise and blur occasionally to improve robustness
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.4),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
        ], p=0.3),
        
        # Edge enhancement (key component)
        EdgeEnhancedTransform(enhance_method=edge_enhancement, strength=0.4, p=0.8),
        
        # Optional gradient enhancement
        GradientEnhancedTransform(gradient_strength=0.2, p=0.5),
        
        # Color space augmentations that preserve edges
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_edge_enhanced_val_transforms(input_size=(384, 384), edge_enhancement='unsharp_mask'):
    """
    Validation transforms with consistent edge enhancement
    """
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        
        # Apply consistent edge enhancement for validation
        EdgeEnhancedTransform(enhance_method=edge_enhancement, strength=0.4, p=1.0),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_boundary_optimized_transforms(input_size=(384, 384)):
    """
    Specialized transforms optimized specifically for boundary detection
    """
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.6),
        
        # Boundary-preserving photometric augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
        # Simple edge enhancement (avoid complex transforms that cause issues)
        EdgeEnhancedTransform(enhance_method='unsharp_mask', strength=0.4, p=0.8),
        
        # Gradient enhancement for boundary clarity
        GradientEnhancedTransform(gradient_strength=0.25, p=0.7),
        
        # Color augmentations that maintain edge contrast
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.6),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Utility function to visualize the effect of edge enhancement
def visualize_edge_enhancement(image_path: str, save_path: Optional[str] = None):
    """
    Visualize the effect of different edge enhancement methods
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Load image
    img = np.array(Image.open(image_path).convert('RGB'))
    
    # Create different enhancements
    methods = ['unsharp_mask', 'laplacian', 'sobel', 'clahe_edges']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Enhanced versions
    for i, method in enumerate(methods):
        transform = EdgeEnhancedTransform(enhance_method=method, strength=0.4, p=1.0)
        enhanced = transform.apply(img)
        axes[i+1].imshow(enhanced)
        axes[i+1].set_title(f'{method.replace("_", " ").title()}')
        axes[i+1].axis('off')
    
    # Gradient enhanced
    grad_transform = GradientEnhancedTransform(gradient_strength=0.3, p=1.0)
    grad_enhanced = grad_transform.apply(img)
    axes[5].imshow(grad_enhanced)
    axes[5].set_title('Gradient Enhanced')
    axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test the transforms
    print("Edge-Enhanced Preprocessing Module loaded successfully!")
    print("Available enhancement methods:")
    print("- unsharp_mask: Enhances edges using unsharp masking")
    print("- laplacian: Uses Laplacian filter for edge enhancement") 
    print("- sobel: Uses Sobel operators for gradient-based enhancement")
    print("- clahe_edges: Combines CLAHE with edge enhancement")
    print("- gradient_enhanced: Enhances based on gradient magnitude")
