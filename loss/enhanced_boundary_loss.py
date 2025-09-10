"""
Enhanced Boundary-Aware Loss for DermoMamba
Advanced loss functions specifically designed for precise boundary segmentation
in medical images, particularly skin lesions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
import cv2


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss that specifically targets boundary accuracy
    """
    def __init__(self, edge_weight=2.0, thickness=3):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.thickness = thickness
        
    def extract_boundary(self, mask, thickness=3):
        """Extract boundary region from mask"""
        mask_np = mask.cpu().numpy()
        boundaries = []
        
        for i in range(mask_np.shape[0]):
            binary_mask = (mask_np[i] > 0.5).astype(np.uint8)
            
            # Create boundary by erosion/dilation
            eroded = binary_erosion(binary_mask, iterations=thickness//2)
            dilated = binary_dilation(binary_mask, iterations=thickness//2)
            boundary = dilated.astype(np.float32) - eroded.astype(np.float32)
            
            boundaries.append(boundary)
        
        boundaries = np.stack(boundaries, axis=0)
        return torch.from_numpy(boundaries).float().to(mask.device)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth [B, 1, H, W] or [B, H, W]
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Extract boundary regions
        boundary_mask = self.extract_boundary(target.squeeze(1), self.thickness)
        
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float())
        
        # Boundary-focused BCE loss
        boundary_pred = pred.squeeze(1) * boundary_mask
        boundary_target = target.squeeze(1).float() * boundary_mask
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_pred, boundary_target, reduction='none'
        )
        boundary_loss = (boundary_bce * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)
        
        return bce_loss + self.edge_weight * boundary_loss


class ContourPreservingLoss(nn.Module):
    """
    Loss function that preserves contour integrity and topology
    """
    def __init__(self, contour_weight=1.5, hausdorff_weight=0.5):
        super(ContourPreservingLoss, self).__init__()
        self.contour_weight = contour_weight
        self.hausdorff_weight = hausdorff_weight
    
    def contour_loss(self, pred, target):
        """Contour-based loss using morphological operations"""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Morphological gradient (edge detection)
        kernel = torch.ones(3, 3).to(pred.device)
        
        # Dilate and erode to get morphological gradient
        pred_dilated = F.max_pool2d(pred_sigmoid, kernel_size=3, stride=1, padding=1)
        pred_eroded = -F.max_pool2d(-pred_sigmoid, kernel_size=3, stride=1, padding=1)
        pred_gradient = pred_dilated - pred_eroded
        
        target_dilated = F.max_pool2d(target.float(), kernel_size=3, stride=1, padding=1)
        target_eroded = -F.max_pool2d(-target.float(), kernel_size=3, stride=1, padding=1)
        target_gradient = target_dilated - target_eroded
        
        return F.mse_loss(pred_gradient, target_gradient)
    
    def hausdorff_loss_approximation(self, pred, target):
        """Approximation of Hausdorff distance for differentiable loss"""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Distance transform approximation
        def soft_distance_transform(x):
            # Use morphological operations to approximate distance transform
            distances = []
            for i in range(1, 8):  # Multiple scales
                eroded = -F.max_pool2d(-x, kernel_size=2*i+1, stride=1, padding=i)
                dist = x - eroded
                distances.append(dist * i)
            return torch.stack(distances).max(dim=0)[0]
        
        pred_dist = soft_distance_transform(pred_sigmoid)
        target_dist = soft_distance_transform(target.float())
        
        return F.l1_loss(pred_dist, target_dist)
    
    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        contour_loss = self.contour_loss(pred, target)
        hausdorff_loss = self.hausdorff_loss_approximation(pred, target)
        
        return self.contour_weight * contour_loss + self.hausdorff_weight * hausdorff_loss


class AdvancedBoundaryLoss(nn.Module):
    """
    Advanced boundary loss combining multiple boundary-aware techniques
    """
    def __init__(self, 
                 dice_weight=1.0,
                 edge_weight=2.0,
                 contour_weight=1.5,
                 topology_weight=1.0,
                 consistency_weight=0.5):
        super(AdvancedBoundaryLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.contour_weight = contour_weight
        self.topology_weight = topology_weight
        self.consistency_weight = consistency_weight
        
        self.edge_loss = EdgeAwareLoss(edge_weight=1.0)
        self.contour_loss = ContourPreservingLoss(contour_weight=1.0)
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Standard Dice loss"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()
    
    def topology_preserving_loss(self, pred, target):
        """Loss that preserves topological properties"""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Betti number approximation using Euler characteristic
        def euler_characteristic_2d(x):
            # Simplified topological loss based on connected components
            # This is an approximation for differentiable computation
            x_binary = (x > 0.5).float()
            
            # Count transitions (approximation of topology)
            horizontal_transitions = torch.abs(x_binary[:, :, :, 1:] - x_binary[:, :, :, :-1]).sum()
            vertical_transitions = torch.abs(x_binary[:, :, 1:, :] - x_binary[:, :, :-1, :]).sum()
            
            return horizontal_transitions + vertical_transitions
        
        pred_topology = euler_characteristic_2d(pred_sigmoid)
        target_topology = euler_characteristic_2d(target.float())
        
        return torch.abs(pred_topology - target_topology) / (target_topology + 1e-6)
    
    def consistency_loss(self, pred, target):
        """Multi-scale consistency loss"""
        losses = []
        scales = [1.0, 0.5, 0.25]
        
        for scale in scales:
            if scale != 1.0:
                size = [int(target.shape[-2] * scale), int(target.shape[-1] * scale)]
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target.float(), size=size, mode='nearest')
                
                # Upscale back to original size
                pred_upscaled = F.interpolate(pred_scaled, size=target.shape[-2:], mode='bilinear', align_corners=False)
                
                consistency = F.mse_loss(torch.sigmoid(pred), torch.sigmoid(pred_upscaled))
                losses.append(consistency)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=pred.device)
    
    def forward(self, pred, target):
        """
        Complete advanced boundary loss
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Component losses
        dice_loss = self.dice_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        contour_loss = self.contour_loss(pred, target)
        topology_loss = self.topology_preserving_loss(pred, target)
        consistency_loss = self.consistency_loss(pred, target)
        
        # Combined loss
        total_loss = (self.dice_weight * dice_loss +
                     self.edge_weight * edge_loss +
                     self.contour_weight * contour_loss +
                     self.topology_weight * topology_loss +
                     self.consistency_weight * consistency_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'dice_loss': dice_loss.item(),
            'edge_loss': edge_loss.item(),
            'contour_loss': contour_loss.item(),
            'topology_loss': topology_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
        
        return total_loss, loss_dict


class BoundaryIoULoss(nn.Module):
    """
    IoU loss specifically computed on boundary regions
    """
    def __init__(self, boundary_thickness=5, smooth=1e-6):
        super(BoundaryIoULoss, self).__init__()
        self.boundary_thickness = boundary_thickness
        self.smooth = smooth
    
    def extract_boundary_region(self, mask, thickness):
        """Extract boundary region using morphological operations"""
        mask_np = mask.cpu().numpy()
        boundaries = []
        
        for i in range(mask_np.shape[0]):
            binary_mask = (mask_np[i] > 0.5).astype(np.uint8)
            
            # Create thick boundary region
            kernel = np.ones((thickness, thickness), np.uint8)
            dilated = cv2.dilate(binary_mask, kernel, iterations=1)
            eroded = cv2.erode(binary_mask, kernel, iterations=1)
            boundary_region = dilated.astype(np.float32) - eroded.astype(np.float32)
            
            # Make boundary region thicker for better gradient flow
            boundary_region = cv2.dilate(boundary_region, kernel, iterations=1)
            boundaries.append(boundary_region)
        
        boundaries = np.stack(boundaries, axis=0)
        return torch.from_numpy(boundaries).float().to(mask.device)
    
    def boundary_iou(self, pred, target, boundary_mask):
        """Compute IoU specifically on boundary regions"""
        pred_sigmoid = torch.sigmoid(pred.squeeze(1))
        target_float = target.squeeze(1).float()
        
        # Focus only on boundary regions
        pred_boundary = pred_sigmoid * boundary_mask
        target_boundary = target_float * boundary_mask
        
        intersection = (pred_boundary * target_boundary).sum(dim=(1, 2))
        union = (pred_boundary + target_boundary - pred_boundary * target_boundary).sum(dim=(1, 2))
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()
    
    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Extract boundary regions
        boundary_mask = self.extract_boundary_region(target.squeeze(1), self.boundary_thickness)
        
        # Compute boundary IoU loss
        boundary_iou_loss = self.boundary_iou(pred, target, boundary_mask)
        
        # Standard IoU loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target.float()).sum(dim=(2, 3))
        union = (pred_sigmoid + target.float() - pred_sigmoid * target.float()).sum(dim=(2, 3))
        standard_iou_loss = 1 - ((intersection + self.smooth) / (union + self.smooth)).mean()
        
        # Combine with higher weight on boundary
        return 0.3 * standard_iou_loss + 0.7 * boundary_iou_loss


# Factory function for creating boundary losses
def create_boundary_loss(loss_type='advanced', **kwargs):
    """
    Create different types of boundary-aware losses
    
    Args:
        loss_type: 'edge', 'contour', 'advanced', 'boundary_iou'
        **kwargs: Additional parameters
    """
    if loss_type == 'edge':
        return EdgeAwareLoss(**kwargs)
    elif loss_type == 'contour':
        return ContourPreservingLoss(**kwargs)
    elif loss_type == 'advanced':
        return AdvancedBoundaryLoss(**kwargs)
    elif loss_type == 'boundary_iou':
        return BoundaryIoULoss(**kwargs)
    else:
        raise ValueError(f"Unknown boundary loss type: {loss_type}")


# Test function
def test_boundary_losses():
    """Test all boundary loss implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size, height, width = 2, 256, 256
    pred = torch.randn(batch_size, 1, height, width).to(device)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)
    
    # Test different loss types
    loss_types = ['edge', 'contour', 'advanced', 'boundary_iou']
    
    print("Boundary Loss Test Results:")
    print("=" * 50)
    
    for loss_type in loss_types:
        try:
            loss_fn = create_boundary_loss(loss_type).to(device)
            
            if loss_type == 'advanced':
                loss_value, loss_dict = loss_fn(pred, target)
                print(f"\n{loss_type.upper()} Loss:")
                print(f"Total: {loss_value.item():.4f}")
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        print(f"  {key}: {value:.4f}")
            else:
                loss_value = loss_fn(pred, target)
                print(f"{loss_type.upper()} Loss: {loss_value.item():.4f}")
                
        except Exception as e:
            print(f"{loss_type.upper()} Loss: Error - {str(e)}")


if __name__ == "__main__":
    test_boundary_losses()
