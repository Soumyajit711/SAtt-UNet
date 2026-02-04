import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid((x - mean) / torch.sqrt(var + self.eps))

class DepthwiseBlock(nn.Module):
    def __init__(self, channels, kernel_size, l2_reg=1e-3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding='same',
            groups=channels, bias=False
        )
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='sigmoid')
    
    def forward(self, x):
        return torch.sigmoid(self.depthwise(x))

class D2M(nn.Module):
    """Depth-to-Mid: 1x1 projection to a fixed decoder width (no spatial change)."""
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.proj(x)

class MSDBlock(nn.Module):
    """Enhanced Multi-Scale Dilated block with residual connections"""
    def __init__(self, ch, dw_groups=None):
        super().__init__()
        g = ch if dw_groups is None else dw_groups
        self.dw1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1, groups=g, bias=False)
        self.dw2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2, groups=g, bias=False)
        self.dw3 = nn.Conv2d(ch, ch, 3, padding=3, dilation=3, groups=g, bias=False)
        self.dw4 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4, groups=g, bias=False)  # Added 4th scale
        
        self.bn = nn.BatchNorm2d(4 * ch)  # Updated for 4 scales
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        identity = self.residual(x)
        
        x1 = self.dw1(x)
        x2 = self.dw2(x)
        x3 = self.dw3(x)
        x4 = self.dw4(x)
        
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_cat = self.bn(x_cat)
        x_fused = self.fuse(x_cat)
        
        # Residual connection
        return F.gelu(x_fused + identity)

class UpConv(nn.Module):
    """Enhanced Up-sampling with feature refinement"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.refine(x)
        return x
