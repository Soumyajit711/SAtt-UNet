import torch
import torch.nn as nn
from .encoder import EnhancedEncoder
from .correlation import CorrelationGuidedCSA
from .blocks import UpConv
from .decoder import EnhancedDecoderMSD

class MSDUNet_MBV2_CorrCSA(nn.Module):
    """Enhanced MSDUNet with improved encoder-decoder architecture"""
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        
        # Enhanced encoder
        self.encoder = EnhancedEncoder(pretrained=pretrained)
        
        # Correlation-guided refinement on skip paths with correct channel sizes
        self.corr0 = CorrelationGuidedCSA(16, num_heads=2)   # e0 has 16 channels
        self.corr1 = CorrelationGuidedCSA(24, num_heads=4)   # e1 has 24 channels
        self.corr2 = CorrelationGuidedCSA(32, num_heads=4)   # e2 has 32 channels
        self.corr3 = CorrelationGuidedCSA(96, num_heads=8)   # e3 has 96 channels
        
        # UpConv from deepest
        self.up_from_bottleneck = UpConv(in_ch=1280, out_ch=96)
        
        # Enhanced decoders with matching channel dimensions
        self.dec2 = EnhancedDecoderMSD(deep_ch=96, skip_ch=32, out_ch=32, use_csa=True, heads=4)
        self.dec1 = EnhancedDecoderMSD(deep_ch=32, skip_ch=24, out_ch=24, use_csa=True, heads=4)
        self.dec0 = EnhancedDecoderMSD(deep_ch=24, skip_ch=16, out_ch=16, use_csa=True, heads=2)
        
        # Final processing
        self.final_up = nn.Sequential(
            UpConv(in_ch=16, out_ch=16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(8, num_classes, 1)
        )
        
    def forward(self, x, return_aux=False):
        # Enhanced encoder
        e0, e1, e2, e3, deep = self.encoder(x)
        
        # UpConv from deepest
        d3 = self.up_from_bottleneck(deep)
        
        # Correlation-guided skip refinement
        s2 = self.corr2(e2)  # e2 has 32 channels
        s1 = self.corr1(e1)  # e1 has 24 channels
        s0 = self.corr0(e0)  # e0 has 16 channels
        
        # Enhanced decoder
        d2 = self.dec2(d3, s2)  # d3:96, s2:32 -> out:32
        d1 = self.dec1(d2, s1)  # d2:32, s1:24 -> out:24
        d0 = self.dec0(d1, s0)  # d1:24, s0:16 -> out:16
        
        # Final output
        out = self.final_up(d0)
        out = self.seg_head(out)
        output = torch.sigmoid(out)
        
        return output
