"""
Comprehensive Spatial-Aware Edge Enhancement
Implements ALL edge methods + spatial data as described in the DermoMamba paper
"""

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

class SixChannelToTensor(A.BasicTransform):
    """Custom transform to convert 6-channel images to tensors"""
    
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
    
    @property
    def targets(self):
        return {"image": self.apply}
    
    def apply(self, img, **params):
        """Convert 6-channel image to tensor"""
        if len(img.shape) == 3 and img.shape[2] == 6:
            # Convert HWC to CHW format
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor
        else:
            # Fall back to regular conversion
            return ToTensorV2()(image=img)['image']

class ComprehensiveSpatialEdgeTransform:
    """
    Comprehensive transform that applies ALL edge enhancement methods
    AND adds spatial coordinate information like the paper
    
    This is a standalone transform, not using albumentations framework
    """
    
    def __init__(self, 
                 input_size=(384, 384),
                 apply_all_methods=True,
                 add_spatial_coords=True,
                 edge_strength=0.7):
        self.input_size = input_size
        self.apply_all_methods = apply_all_methods
        self.add_spatial_coords = add_spatial_coords
        self.edge_strength = edge_strength
    
    def __call__(self, image):
        """Make the transform callable directly"""
        return self.apply_transform(image)
    
    def apply_transform(self, img):
        """Apply comprehensive edge enhancement to image"""
        
        if not self.apply_all_methods:
            return img
            
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Method 1: Unsharp Masking
        enhanced_1 = self._unsharp_mask(img_float)
        
        # Method 2: CLAHE + Edge Enhancement
        enhanced_2 = self._clahe_edge_enhancement(img_float)
        
        # Method 3: Sobel Edge Enhancement
        enhanced_3 = self._sobel_enhancement(img_float)
        
        # Method 4: Laplacian Enhancement
        enhanced_4 = self._laplacian_enhancement(img_float)
        
        # Method 5: Multi-scale Enhancement
        enhanced_5 = self._multiscale_edge_enhancement(img_float)
        
        # Combine all enhancement methods
        combined_enhanced = (
            0.3 * enhanced_1 + 
            0.25 * enhanced_2 + 
            0.2 * enhanced_3 + 
            0.15 * enhanced_4 + 
            0.1 * enhanced_5
        )
        
        # Clip to valid range
        combined_enhanced = np.clip(combined_enhanced, 0, 1)
        
        # Convert back to uint8 for spatial coordinate addition
        final_enhanced = (combined_enhanced * 255).astype(np.uint8)
        
        # Add spatial coordinates to create 6-channel output
        if self.add_spatial_coords:
            final_output = self._add_spatial_coordinates(final_enhanced)
            return final_output  # Returns (H, W, 6) array
        else:
            return final_enhanced  # Returns (H, W, 3) array
        
        # Method 5: Multi-scale Edge Features
        enhanced_5 = self._multiscale_edge_enhancement(img_float)
        
        # Combine all enhancements with weighted fusion
        weights = [0.25, 0.2, 0.2, 0.15, 0.2]  # Emphasize unsharp mask slightly
        combined = (weights[0] * enhanced_1 + 
                   weights[1] * enhanced_2 + 
                   weights[2] * enhanced_3 + 
                   weights[3] * enhanced_4 + 
                   weights[4] * enhanced_5)
        
        # Apply edge strength control
        final = img_float + self.edge_strength * (combined - img_float)
        
        # Clip and convert back
        final = np.clip(final * 255, 0, 255).astype(np.uint8)
        
        # Add spatial coordinate channels if requested
        if self.add_spatial_coords:
            final = self._add_spatial_coordinates(final)
        
        return final
    
    def apply_to_mask(self, mask, **params):
        """Apply to mask (no modification needed for spatial coords)"""
        return mask
    
    def _unsharp_mask(self, img):
        """Enhanced unsharp masking"""
        blurred = cv2.GaussianBlur(img, (9, 9), 2.0)
        return img + 0.8 * (img - blurred)
    
    def _clahe_edge_enhancement(self, img):
        """CLAHE with edge enhancement"""
        # Convert to LAB for better contrast enhancement
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_float = lab.astype(np.float32)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab_float[:,:,0] = clahe.apply((lab_float[:,:,0]).astype(np.uint8)).astype(np.float32)
            
            # Convert back
            enhanced = cv2.cvtColor(lab_float.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply((img * 255).astype(np.uint8)).astype(np.float32) / 255.0
        
        return enhanced
    
    def _sobel_enhancement(self, img):
        """Sobel edge enhancement"""
        if len(img.shape) == 3:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
        
        # Compute Sobel edges - convert to uint8 for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)
        sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        sobel_norm = (sobel_combined - sobel_combined.min()) / (sobel_combined.max() - sobel_combined.min() + 1e-8)
        
        # Apply edge enhancement
        if len(img.shape) == 3:
            # Add edge information to all channels
            enhanced = img.copy()
            for c in range(3):
                enhanced[:,:,c] = img[:,:,c] + 0.3 * sobel_norm
        else:
            enhanced = img + 0.3 * sobel_norm
        
        return enhanced
    
    def _laplacian_enhancement(self, img):
        """Laplacian edge enhancement"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
        
        # Laplacian edge detection - convert to uint8 for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F, ksize=3)
        laplacian_norm = np.abs(laplacian)
        laplacian_norm = (laplacian_norm - laplacian_norm.min()) / (laplacian_norm.max() - laplacian_norm.min() + 1e-8)
        
        # Apply enhancement
        if len(img.shape) == 3:
            enhanced = img.copy()
            for c in range(3):
                enhanced[:,:,c] = img[:,:,c] + 0.2 * laplacian_norm
        else:
            enhanced = img + 0.2 * laplacian_norm
        
        return enhanced
    
    def _multiscale_edge_enhancement(self, img):
        """Multi-scale edge features at different scales"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
        
        # Different scales
        scales = [1, 2, 4]
        edge_maps = []
        
        for scale in scales:
            # Blur at different scales
            sigma = scale * 0.8
            gray_uint8 = (gray * 255).astype(np.uint8)
            # Ensure odd kernel size
            ksize = max(3, int(6*sigma+1))
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(gray_uint8, (ksize, ksize), sigma)
            
            # Edge detection at this scale
            edges = cv2.Canny(blurred, 50, 150)
            edge_maps.append(edges.astype(np.float32) / 255.0)
        
        # Combine multi-scale edges
        combined_edges = np.mean(edge_maps, axis=0)
        
        # Apply to image
        if len(img.shape) == 3:
            enhanced = img.copy()
            for c in range(3):
                enhanced[:,:,c] = img[:,:,c] + 0.4 * combined_edges
        else:
            enhanced = img + 0.4 * combined_edges
        
        return enhanced
    
    def _add_spatial_coordinates(self, img):
        """Add spatial coordinate information as additional channels (like the paper)"""
        # Use actual image dimensions, not self.input_size
        if len(img.shape) == 3:
            h, w = img.shape[:2]
        else:
            h, w = img.shape
        
        # Create coordinate grids
        x_coords = np.linspace(-1, 1, w)
        y_coords = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Radial distance from center
        rr = np.sqrt(xx**2 + yy**2)
        
        # Normalize coordinates
        xx_norm = (xx + 1) * 127.5  # Scale to 0-255
        yy_norm = (yy + 1) * 127.5  # Scale to 0-255
        rr_norm = (rr / rr.max()) * 255  # Scale to 0-255
        
        # Stack with image (now 6 channels: RGB + X + Y + R)
        if len(img.shape) == 3:
            # RGB image
            spatial_enhanced = np.stack([
                img[:,:,0], img[:,:,1], img[:,:,2],  # Original RGB
                xx_norm.astype(np.uint8),            # X coordinates  
                yy_norm.astype(np.uint8),            # Y coordinates
                rr_norm.astype(np.uint8)             # Radial distance
            ], axis=2)
        else:
            # Grayscale image - convert to RGB first, then add spatial
            spatial_enhanced = np.stack([
                img, img, img,                       # Convert grayscale to RGB
                xx_norm.astype(np.uint8),            # X coordinates
                yy_norm.astype(np.uint8),            # Y coordinates  
                rr_norm.astype(np.uint8)             # Radial distance
            ], axis=2)
        
        return spatial_enhanced

class AlbumentationsComprehensiveTransform(A.ImageOnlyTransform):
    """Albumentations-compatible wrapper for ComprehensiveSpatialEdgeTransform"""
    
    def __init__(self, input_size=(384, 384), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.transform = ComprehensiveSpatialEdgeTransform(input_size=input_size)
    
    def apply(self, img, **params):
        return self.transform.apply_transform(img)

def get_comprehensive_spatial_train_transforms(input_size=(384, 384)):
    """Training transforms with comprehensive spatial-aware edge enhancement"""
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        
        # Comprehensive spatial-aware edge enhancement
        AlbumentationsComprehensiveTransform(input_size=input_size),
        
        # Standard augmentations (but be careful with 6-channel images)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        
        # Skip transformations that might not handle 6 channels well
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # A.GaussNoise(p=0.2),
        
        # Convert to tensor (handles 6 channels)
        SixChannelToTensor()
    ])

def get_comprehensive_spatial_val_transforms(input_size=(384, 384)):
    """Validation transforms with comprehensive spatial-aware edge enhancement"""  
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        
        # Comprehensive spatial-aware edge enhancement
        AlbumentationsComprehensiveTransform(input_size=input_size),
        
        # Convert to tensor (handles 6 channels)
        SixChannelToTensor()
    ])
