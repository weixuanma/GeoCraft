import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAChannelAttention(nn.Module):
    """Enhanced Channel Attention (ECA): No dimensionality reduction, local cross-channel interaction"""
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # Global average pooling
        feat = self.pool(x).view(B, 1, C)  # [B, 1, C]
        # 1D convolution for local cross-channel interaction
        feat = self.conv(feat)  # [B, 1, C]
        # Channel attention weight
        attn = self.sigmoid(feat).view(B, C, 1, 1)  # [B, C, 1, 1]
        # Apply attention
        return x * attn

class DynamicFilter(nn.Module):
    """Dynamic Filter (DF): Generate filter parameters based on local spatial features"""
    def __init__(self, in_channels, filter_size=3):
        super().__init__()
        self.filter_size = filter_size
        self.filter_params = nn.Conv2d(in_channels, in_channels * filter_size**2, 1)
        self.padding = (filter_size - 1) // 2

    def forward(self, x):
        B, C, H, W = x.shape
        # Generate filter parameters: [B, C*K², H, W] (K=filter_size)
        filters = self.filter_params(x)
        filters = filters.view(B, C, self.filter_size**2, H, W)  # [B, C, K², H, W]
        filters = filters.permute(0, 1, 3, 4, 2)  # [B, C, H, W, K²]
        
        # Unfold input to local patches: [B, C*K², H, W]
        x_unfolded = F.unfold(x, kernel_size=self.filter_size, padding=self.padding).view(B, C, self.filter_size**2, H, W)
        x_unfolded = x_unfolded.permute(0, 1, 3, 4, 2)  # [B, C, H, W, K²]
        
        # Apply dynamic filter (element-wise multiply + sum over K²)
        filtered_feat = torch.sum(x_unfolded * filters, dim=-1)  # [B, C, H, W]
        return filtered_feat

class DDFModule(nn.Module):
    """Dynamic Dual Fusion (DDF) Module: ECA + DF for channel-spatial dynamic fusion"""
    def __init__(self, in_channels, eca_kernel_size=3, dynamic_filter_size=3, fusion_ratio=0.5):
        super().__init__()
        self.fusion_ratio = fusion_ratio
        
        # Channel-wise dynamic fusion (ECA)
        self.eca_attention = ECAChannelAttention(in_channels, eca_kernel_size)
        
        # Spatial-wise dynamic fusion (DF)
        self.dynamic_filter = DynamicFilter(in_channels, dynamic_filter_size)
        
        # Feature projection for consistency
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Channel-wise enhanced feature
        channel_feat = self.eca_attention(x)
        
        # Spatial-wise enhanced feature
        spatial_feat = self.dynamic_filter(x)
        
        # Dual fusion (weighted sum)
        fused_feat = self.fusion_ratio * channel_feat + (1 - self.fusion_ratio) * spatial_feat
        
        # Post-processing
        fused_feat = self.proj(fused_feat)
        fused_feat = self.batch_norm(fused_feat)
        
        return fused_feat