import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=2.0, smooth=1e-6):
        super(AdaptiveSegmentationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
        # Ensure weights sum to 1
        assert alpha + beta <= 1.0, "alpha + beta must be <= 1"
        
    def forward(self, y_pred, y_true):
        # Handle multi-scale outputs
        if isinstance(y_pred, tuple):
            main_output, aux1, aux2 = y_pred
            
            # Resize auxiliary outputs to match target
            aux1_resized = F.interpolate(aux1, size=y_true.shape[-2:], mode='bilinear', align_corners=False)
            aux2_resized = F.interpolate(aux2, size=y_true.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute losses
            main_loss = self._compute_loss(main_output, y_true)
            aux1_loss = self._compute_loss(aux1_resized, y_true)
            aux2_loss = self._compute_loss(aux2_resized, y_true)
            
            # Weighted combination
            total_loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss
            return total_loss
        else:
            return self._compute_loss(y_pred, y_true)
    
    def _compute_loss(self, y_pred, y_true):
        # Dice Loss
        intersection = (y_pred * y_true).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth)
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        
        # Focal Loss
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce)  # prevents nans when pt=0
        focal_loss = ((1 - pt) ** self.gamma * bce).mean()
        
        # Combined loss
        loss = (self.alpha * dice_loss + 
                (1 - self.alpha - self.beta) * bce_loss + 
                self.beta * focal_loss)
        
        return loss
