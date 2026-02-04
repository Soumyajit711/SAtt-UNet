import torch
import torch.nn as nn
from .blocks import DepthwiseBlock, SimAM

class CSAHead(nn.Module):
    def __init__(self, channels, dropout=0.1, l2_reg=1e-3):
        super().__init__()
        self.blocks = nn.ModuleList([
            DepthwiseBlock(channels, (1, 1), l2_reg),
            DepthwiseBlock(channels, (1, 2), l2_reg),
            DepthwiseBlock(channels, (2, 1), l2_reg),
            DepthwiseBlock(channels, (1, 3), l2_reg),
            DepthwiseBlock(channels, (3, 1), l2_reg),
            DepthwiseBlock(channels, (1, 4), l2_reg),
            DepthwiseBlock(channels, (4, 1), l2_reg),
            DepthwiseBlock(channels, (1, 5), l2_reg),
            DepthwiseBlock(channels, (5, 1), l2_reg)
        ])
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(channels)
        
    def sequential_attention(self, x):
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x1)
        x = x2 * x3
        x4 = self.blocks[3](x)
        x5 = self.blocks[4](x)
        x = x4 * x5
        x6 = self.blocks[5](x)
        x7 = self.blocks[6](x)
        x = x6 * x7
        x8 = self.blocks[7](x)
        x9 = self.blocks[8](x)
        x = x8 * x9
        return self.dropout(x)

    def row_column_attention(self, x):
        row = self.blocks[1](x)
        for k in [3, 5, 7]:
            row = self.blocks[k](row)
        
        col = self.blocks[2](x)
        for k in [4, 6, 8]:
            col = self.blocks[k](col)
        
        row = self.dropout(row)
        col = self.dropout(col)
        return row * col

    def forward(self, x):
        x1 = self.sequential_attention(x)
        rc_attn = self.row_column_attention(x1)
        final = x1 + rc_attn
        final = self.dropout(self.bn(final))
        return final

class CSA(nn.Module):
    def __init__(self, num_heads, channels, dropout_rate=0.1, l2_reg=1e-3):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.output_channels = channels
        
        # Heads process individual channel groups
        self.heads = nn.ModuleList([
            CSAHead(self.head_channels, dropout=dropout_rate, l2_reg=l2_reg)
            for _ in range(num_heads)
        ])
        
        # SIMAM attention
        self.simam = SimAM()
        
        # Channel projection layers
        self.heads_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.simam_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Final processing
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm2d(channels)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Process through heads
        head_outputs = []
        for i in range(self.num_heads):
            # Split channels for each head
            x_slice = x[:, i*self.head_channels:(i+1)*self.head_channels, :, :]
            out = self.heads[i](x_slice)
            head_outputs.append(out)
        
        # Combine head outputs
        cbam_out = torch.cat(head_outputs, dim=1)
        cbam_out = self.heads_proj(cbam_out)
        
        # Process through SIMAM
        simam_out = self.simam(x)
        simam_out = self.simam_proj(simam_out)
        
        # Combine features
        fused = cbam_out + simam_out
        fused = self.dropout(fused)
        fused = self.batch_norm(fused)
        
        return fused

class PearsonCorrelationLayer(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        B, C, H, W = x.shape
        f = x.view(B, C, -1)
        f = f - f.mean(dim=-1, keepdim=True)
        cov = torch.bmm(f, f.transpose(1, 2)) / (f.size(-1) - 1 + self.eps)
        var = torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(-1)
        denom = torch.sqrt(var @ var.transpose(1, 2) + self.eps)
        return cov / denom

class SpearmanCorrelationLayer(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        B, C, H, W = x.shape
        f = x.view(B, C, -1)
        ranks = torch.argsort(torch.argsort(f, dim=-1), dim=-1).float()
        ranks = ranks - ranks.mean(dim=-1, keepdim=True)
        cov = torch.bmm(ranks, ranks.transpose(1, 2)) / (ranks.size(-1) - 1 + self.eps)
        var = torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(-1)
        denom = torch.sqrt(var @ var.transpose(1, 2) + self.eps)
        return cov / denom

class CorrelationGuidedCSA(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.csa = CSA(num_heads=num_heads, channels=channels)
        
        # Simplified correlation guidance
        self.corr_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Get channel-wise attention based on global context
        corr_attn = self.corr_attention(x)
        
        # Enhanced feature modulation
        x_mod = x * corr_attn
        return self.csa(x_mod)
