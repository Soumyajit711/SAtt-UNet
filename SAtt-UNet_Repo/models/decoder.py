import torch
import torch.nn as nn
from .blocks import MSDBlock, UpConv, D2M
from .correlation import CSA

class EnhancedDecoderMSD(nn.Module):
    """Enhanced MSDUNet decoder with improved feature processing and attention"""
    def __init__(self, deep_ch, skip_ch, out_ch, use_csa=True, heads=4):
        super().__init__()
        self.up = UpConv(deep_ch, out_ch)
        self.d2m = D2M(skip_ch, out_ch)  # This should project skip_ch to out_ch
        self.msd = MSDBlock(out_ch)
        
        # Enhanced fusion with cross-attention
        self.cross_attention = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True) if out_ch >= 32 else None
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        
        self.use_csa = use_csa
        self.csa = CSA(num_heads=heads, channels=out_ch) if use_csa else nn.Identity()
        
        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(out_ch // 4, 1), 1),  # Ensure at least 1 channel
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_ch // 4, 1), out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, deep, skip):
        d = self.up(deep)
        s = self.msd(self.d2m(skip))  # Project skip to out_ch and process
        
        # Cross-attention between deep and skip features
        if self.cross_attention is not None and d.shape[1] >= 32:
            B, C, H, W = d.shape
            d_flat = d.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
            s_flat = s.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
            
            d_attended, _ = self.cross_attention(d_flat, s_flat, s_flat)
            d = d_attended.transpose(1, 2).view(B, C, H, W)
        
        # Enhanced fusion
        fused = torch.cat([d, s], dim=1)
        fused = self.fusion_conv(fused)
        
        # Residual connection with SE attention
        se_weight = self.se(fused)
        x = d + fused * se_weight
        
        x = self.fuse(x)
        x = self.csa(x)
        return x
