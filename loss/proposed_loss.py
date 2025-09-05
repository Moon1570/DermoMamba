from scipy.ndimage import distance_transform_edt
import torch
from loss.loss import DiceLoss
import torch.nn.functional as F

def compute_attention_mask_boundary(binary_mask):
    # Convert PyTorch tensor to numpy if needed
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.numpy()
    
    # Ensure binary mask is properly formatted (values are 0 or 1)
    binary_mask = (binary_mask > 0.5).astype(int)
    
    dist_foreground = distance_transform_edt(binary_mask == 0)
    dist_background = distance_transform_edt(binary_mask == 1)
    distance_map = dist_foreground + dist_background
    attention_mask = 1 / (1 + distance_map)
    return torch.tensor(attention_mask, dtype=torch.float32)
def Guide_Fusion_Loss(y_pred, y_true):
    # Ensure shapes match for loss calculation
    if y_true.ndim == 3 and y_pred.ndim == 4:
        y_true = y_true.unsqueeze(1)
    
    dice = DiceLoss()(y_pred, y_true)
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
    
    # Handle CPU tensors and convert y_true to correct format for attention mask
    y_true_np = y_true.squeeze(1).cpu() if y_true.ndim == 4 else y_true.cpu()
    
    if y_true.is_cuda:
        attention_mask = compute_attention_mask_boundary(y_true_np).cuda()
    else:
        attention_mask = compute_attention_mask_boundary(y_true_np)
    
    loss = bce + dice * attention_mask
    return loss.mean()