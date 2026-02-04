import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class EnhancedEncoder(nn.Module):
    """Enhanced encoder with FPN-like feature enhancement"""
    def __init__(self, pretrained=True):
        super().__init__()
        mb2 = mobilenet_v2(pretrained=pretrained)
        feats = mb2.features

        # Encoder stages
        self.enc0 = nn.Sequential(*feats[:2])     # 16 ch,  H/2
        self.enc1 = nn.Sequential(*feats[2:4])    # 24 ch,  H/4
        self.enc2 = nn.Sequential(*feats[4:7])    # 32 ch,  H/8
        self.enc3 = nn.Sequential(*feats[7:14])   # 96 ch,  H/16
        self.bottleneck = nn.Sequential(*feats[14:])  # 1280 ch, H/32

        # Feature enhancement - keep original channel dimensions
        self.enhance0 = self._make_enhancement_block(16)
        self.enhance1 = self._make_enhancement_block(24)
        self.enhance2 = self._make_enhancement_block(32)
        self.enhance3 = self._make_enhancement_block(96)

    def _make_enhancement_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Bottom-up
        e0 = self.enc0(x)
        e0_enhanced = F.relu(self.enhance0(e0) + e0)

        e1 = self.enc1(e0_enhanced)
        e1_enhanced = F.relu(self.enhance1(e1) + e1)

        e2 = self.enc2(e1_enhanced)
        e2_enhanced = F.relu(self.enhance2(e2) + e2)

        e3 = self.enc3(e2_enhanced)
        e3_enhanced = F.relu(self.enhance3(e3) + e3)

        deep = self.bottleneck(e3_enhanced)

        return e0_enhanced, e1_enhanced, e2_enhanced, e3_enhanced, deep
