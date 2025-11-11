import torch
import torch.nn as nn
import torch.nn.functional as F

class VCMBlock(nn.Module):
    """Visual Cue Fusion Module: Integrate multi-scale visual cues for crack feature enhancement"""
    def __init__(self, in_dim, out_dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.scale_convs = nn.ModuleList()
        self.attention_convs = nn.ModuleList()
        
        # Conv layers for different scales
        for scale in scales:
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim//len(scales), 3, padding=1),
                    nn.BatchNorm2d(out_dim//len(scales)),
                    nn.ReLU()
                )
            )
        
        # Channel attention for each scale
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(out_dim//4, out_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for fused features
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale_feats = []
        B, C, H, W = x.shape
        
        # Extract features at different scales
        for idx, scale in enumerate(self.scales):
            if scale == 1:
                feat = x
            else:
                # Upsample/downsample to target scale
                feat = F.interpolate(
                    x, 
                    size=(H//scale, W//scale), 
                    mode='bilinear', 
                    align_corners=True
                )
            # Process scale-specific feature
            feat = self.scale_convs[idx](feat)
            # Resize back to original size
            feat = F.interpolate(
                feat, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            )
            scale_feats.append(feat)
        
        # Concatenate multi-scale features
        fused_feat = torch.cat(scale_feats, dim=1)
        
        # Apply channel attention
        channel_attn = self.channel_attention(fused_feat)
        fused_feat = fused_feat * channel_attn
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(fused_feat)
        fused_feat = fused_feat * spatial_attn
        
        return fused_feat