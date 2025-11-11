import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """Detection Head: Responsible for crack classification and bounding box regression"""
    def __init__(self, in_dim, num_classes=1, num_conv_layers=2, bbox_reg_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_reg_weight = bbox_reg_weight
        
        # Convolution layers for feature refinement
        self.conv_layers = nn.Sequential()
        for i in range(num_conv_layers):
            self.conv_layers.add_module(
                f'conv_{i}',
                nn.Conv2d(in_dim if i == 0 else in_dim//2, in_dim//2, 3, padding=1)
            )
            self.conv_layers.add_module(f'batch_norm_{i}', nn.BatchNorm2d(in_dim//2))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
        
        # Classification branch (crack/non-crack)
        self.cls_head = nn.Conv2d(in_dim//2, num_classes, 1)
        # Regression branch (bounding box: x1, y1, x2, y2)
        self.reg_head = nn.Conv2d(in_dim//2, 4, 1)
        
        # Loss functions
        self.cls_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification for crack detection
        self.reg_loss_fn = nn.IoULoss(reduction='mean')

    def forward(self, x, targets=None):
        # Feature refinement
        x = self.conv_layers(x)  # [B, C//2, H, W]
        
        # Classification output (logits)
        cls_logits = self.cls_head(x)  # [B, num_classes, H, W]
        # Regression output (bounding box coordinates, normalized to [0,1])
        reg_outputs = torch.sigmoid(self.reg_head(x))  # [B, 4, H, W]
        
        # Prepare outputs dict
        outputs = {
            'cls_logits': cls_logits,
            'reg_outputs': reg_outputs
        }
        
        # Calculate loss if targets are provided (training phase)
        if targets is not None:
            cls_targets, reg_targets = targets  # cls_targets: [B, 1, H, W]; reg_targets: [B, 4, H, W]
            
            # Classification loss
            cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
            
            # Regression loss (only calculate for regions with cracks)
            mask = (cls_targets == 1).float()  # [B, 1, H, W]
            reg_loss = self.reg_loss_fn(reg_outputs * mask, reg_targets * mask) * self.bbox_reg_weight
            
            # Total loss
            total_loss = cls_loss + reg_loss
            outputs['cls_loss'] = cls_loss
            outputs['reg_loss'] = reg_loss
            outputs['total_loss'] = total_loss
        
        return outputs