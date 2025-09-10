"""
DermoMamba Guide Fusion Loss Implementation
Based on the paper: "DermoMamba: a cross-scale Mamba-based model with guide fusion loss 
for skin lesion segmentation in dermoscopy images"

This implementation includes:
1. Boundary-guided attention mechanism
2. Multi-scale boundary detection
3. Adaptive loss weighting
4. Edge-preserving regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, sobel
import cv2


class BoundaryGuidedAttention(nn.Module):
    """
    Boundary-guided attention module that computes attention maps
    based on distance from boundaries and edge information
    """
    def __init__(self, sigma=2.0, boundary_weight=2.0):
        super(BoundaryGuidedAttention, self).__init__()
        self.sigma = sigma
        self.boundary_weight = boundary_weight
    
    def compute_distance_map(self, mask):
        """Compute distance transform from boundaries"""
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        distance_maps = []
        
        for i in range(batch_size):
            binary_mask = (mask_np[i] > 0.5).astype(np.uint8)
            
            # Distance from foreground boundaries
            dist_fg = distance_transform_edt(binary_mask == 0)
            # Distance from background boundaries  
            dist_bg = distance_transform_edt(binary_mask == 1)
            
            # Combined distance map
            distance_map = np.minimum(dist_fg, dist_bg)
            distance_maps.append(distance_map)
        
        distance_maps = np.stack(distance_maps, axis=0)
        return torch.from_numpy(distance_maps).float().to(mask.device)
    
    def compute_edge_map(self, mask):
        """Compute edge map using Sobel operators"""
        mask_np = mask.cpu().numpy()
        batch_size = mask_np.shape[0]
        edge_maps = []
        
        for i in range(batch_size):
            # Sobel edge detection
            sobel_x = sobel(mask_np[i], axis=1)
            sobel_y = sobel(mask_np[i], axis=0)
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_maps.append(edge_map)
        
        edge_maps = np.stack(edge_maps, axis=0)
        return torch.from_numpy(edge_maps).float().to(mask.device)
    
    def forward(self, mask):
        """
        Generate boundary-guided attention map
        Args:
            mask: Ground truth mask [B, H, W] or [B, 1, H, W]
        Returns:
            attention_map: Boundary-guided attention weights [B, H, W]
        """
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        
        # Distance-based attention
        distance_map = self.compute_distance_map(mask)
        distance_attention = torch.exp(-distance_map / self.sigma)
        
        # Edge-based attention
        edge_map = self.compute_edge_map(mask)
        edge_attention = torch.sigmoid(edge_map * self.boundary_weight)
        
        # Combined attention map
        attention_map = distance_attention + edge_attention
        attention_map = torch.clamp(attention_map, min=0.1, max=2.0)  # Prevent extreme values
        
        return attention_map


class MultiScaleBoundaryLoss(nn.Module):
    """
    Multi-scale boundary loss that operates at different resolutions
    to capture both fine and coarse boundary details
    """
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super(MultiScaleBoundaryLoss, self).__init__()
        self.scales = scales
        self.weights = weights
        
    def compute_boundary_loss(self, pred, target, scale=1.0):
        """Compute boundary loss at specific scale"""
        if scale != 1.0:
            size = [int(target.shape[-2] * scale), int(target.shape[-1] * scale)]
            pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target.float(), size=size, mode='nearest')
        else:
            pred_scaled = pred
            target_scaled = target.float()
        
        # Compute gradients (boundaries)
        pred_grad_x = torch.abs(pred_scaled[:, :, :-1, :] - pred_scaled[:, :, 1:, :])
        pred_grad_y = torch.abs(pred_scaled[:, :, :, :-1] - pred_scaled[:, :, :, 1:])
        
        target_grad_x = torch.abs(target_scaled[:, :, :-1, :] - target_scaled[:, :, 1:, :])
        target_grad_y = torch.abs(target_scaled[:, :, :, :-1] - target_scaled[:, :, :, 1:])
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2
    
    def forward(self, pred, target):
        """Multi-scale boundary loss computation"""
        total_loss = 0
        for scale, weight in zip(self.scales, self.weights):
            boundary_loss = self.compute_boundary_loss(pred, target, scale)
            total_loss += weight * boundary_loss
        
        return total_loss


class GuideFusionLoss(nn.Module):
    """
    Complete Guide Fusion Loss as proposed in DermoMamba paper
    Combines Dice loss, BCE loss, boundary guidance, and multi-scale features
    """
    def __init__(self, 
                 dice_weight=1.0,
                 bce_weight=1.0, 
                 boundary_weight=2.0,
                 attention_weight=1.5,
                 multiscale_weight=0.5,
                 focal_gamma=2.0,
                 use_focal=True):
        super(GuideFusionLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.attention_weight = attention_weight
        self.multiscale_weight = multiscale_weight
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        
        # Components
        self.boundary_attention = BoundaryGuidedAttention()
        self.multiscale_boundary = MultiScaleBoundaryLoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Soft Dice Loss"""
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        
        return 1 - dice.mean()
    
    def focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        """Focal Loss for handling class imbalance"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def boundary_enhanced_loss(self, pred, target, attention_map):
        """Boundary-enhanced loss with attention weighting"""
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply attention weighting - higher weight near boundaries
        if attention_map.dim() == 3:
            attention_map = attention_map.unsqueeze(1)
        
        weighted_bce = bce_loss * attention_map
        return weighted_bce.mean()
    
    def forward(self, pred, target):
        """
        Complete Guide Fusion Loss computation
        Args:
            pred: Predicted segmentation logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W] or [B, H, W]
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Ensure target has correct shape
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Generate boundary-guided attention map
        attention_map = self.boundary_attention(target)
        
        # 1. Dice Loss
        dice_loss = self.dice_loss(pred, target)
        
        # 2. Boundary Cross-Entropy or Focal Loss
        if self.use_focal:
            bce_loss = self.focal_loss(pred, target, self.focal_gamma)
        else:
            bce_loss = self.boundary_enhanced_loss(pred, target, attention_map)
        
        # 3. Multi-scale Boundary Loss
        multiscale_loss = self.multiscale_boundary(pred, target)
        
        # 4. Attention-guided loss
        attention_enhanced_dice = self.dice_loss(pred * attention_map.unsqueeze(1), target * attention_map.unsqueeze(1))
        
        # Combine all losses
        total_loss = (self.dice_weight * dice_loss + 
                     self.bce_weight * bce_loss +
                     self.boundary_weight * multiscale_loss +
                     self.attention_weight * attention_enhanced_dice)
        
        # Loss components for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'dice_loss': dice_loss.item(),
            'bce_loss': bce_loss.item(),
            'boundary_loss': multiscale_loss.item(),
            'attention_dice': attention_enhanced_dice.item()
        }
        
        return total_loss, loss_dict


class AdaptiveGuideFusionLoss(GuideFusionLoss):
    """
    Adaptive version that adjusts loss weights based on training progress
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0
        self.initial_weights = {
            'dice': self.dice_weight,
            'bce': self.bce_weight,
            'boundary': self.boundary_weight,
            'attention': self.attention_weight
        }
    
    def update_epoch(self, epoch):
        """Update loss weights based on training epoch"""
        self.epoch = epoch
        
        # Gradually increase boundary and attention focus
        boundary_factor = min(1.0 + epoch * 0.1, 2.0)
        attention_factor = min(1.0 + epoch * 0.05, 1.5)
        
        self.boundary_weight = self.initial_weights['boundary'] * boundary_factor
        self.attention_weight = self.initial_weights['attention'] * attention_factor


# Convenience functions
def create_guide_fusion_loss(loss_type='standard', **kwargs):
    """
    Factory function to create different variants of Guide Fusion Loss
    
    Args:
        loss_type: 'standard', 'adaptive', or 'lightweight'
        **kwargs: Additional parameters for loss configuration
    
    Returns:
        Loss function instance
    """
    if loss_type == 'standard':
        return GuideFusionLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveGuideFusionLoss(**kwargs)
    elif loss_type == 'lightweight':
        # Simplified version with reduced computational cost
        kwargs.update({
            'multiscale_weight': 0.2,
            'boundary_weight': 1.0,
            'use_focal': False
        })
        return GuideFusionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Test function
def test_guide_fusion_loss():
    """Test the Guide Fusion Loss implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size, height, width = 2, 256, 256
    pred = torch.randn(batch_size, 1, height, width).to(device)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)
    
    # Test standard loss
    loss_fn = create_guide_fusion_loss('standard')
    loss_fn = loss_fn.to(device)
    
    total_loss, loss_dict = loss_fn(pred, target)
    
    print("Guide Fusion Loss Test Results:")
    print(f"Total Loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"{key}: {value:.4f}")
    
    # Test adaptive loss
    adaptive_loss_fn = create_guide_fusion_loss('adaptive')
    adaptive_loss_fn = adaptive_loss_fn.to(device)
    adaptive_loss_fn.update_epoch(10)
    
    adaptive_total_loss, adaptive_loss_dict = adaptive_loss_fn(pred, target)
    print(f"\nAdaptive Loss (epoch 10): {adaptive_total_loss.item():.4f}")


if __name__ == "__main__":
    test_guide_fusion_loss()
