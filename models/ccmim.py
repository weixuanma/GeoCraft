import torch
import torch.nn as nn
from models.backbone.mim_block import MiMBlock
from models.backbone.vcm_block import VCMBlock
from models.neck.spt_module import SPTModule
from models.fusion.ddf_module import DDFModule
from models.head.detection_head import DetectionHead

class CCMIM(nn.Module):
    """Concrete Crack Mamba-in-Mamba (CCMIM): End-to-end concrete defect detection framework"""
    def __init__(self, mim_hidden_dim=256, ddf_channel=256, spt_token_level=3, num_classes=1):
        super().__init__()
        # Backbone: MiM + VCM
        self.backbone = nn.Sequential(
            # Initial convolution to map RGB to target dim
            nn.Conv2d(3, mim_hidden_dim//2, 7, stride=2, padding=3),
            nn.BatchNorm2d(mim_hidden_dim//2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # MiM Blocks (hierarchical global-local feature capture)
            MiMBlock(in_dim=mim_hidden_dim//2, hidden_dim=mim_hidden_dim),
            MiMBlock(in_dim=mim_hidden_dim, hidden_dim=mim_hidden_dim),
            
            # VCM Block (multi-scale visual cue fusion)
            VCMBlock(in_dim=mim_hidden_dim, out_dim=mim_hidden_dim)
        )
        
        # Feature Fusion: DDF Module
        self.ddf_fusion = DDFModule(in_channels=mim_hidden_dim)
        
        # Neck: SPT Module
        self.neck = SPTModule(
            in_dim=mim_hidden_dim,
            token_levels=spt_token_level,
            num_heads=4,
            sparse_ratio=0.6
        )
        
        # Detection Head
        self.head = DetectionHead(
            in_dim=mim_hidden_dim,
            num_classes=num_classes,
            num_conv_layers=2,
            bbox_reg_weight=5.0
        )

    def forward(self, x, targets=None):
        # Backbone: Extract multi-scale features
        backbone_feat = self.backbone(x)  # [B, C, H, W]
        
        # Dynamic feature fusion (DDF)
        fused_feat = self.ddf_fusion(backbone_feat)
        
        # Neck: Sparse pyramid attention refinement
        neck_feat = self.neck(fused_feat)
        
        # Detection head: Classification + Regression
        outputs = self.head(neck_feat, targets)
        
        return outputs