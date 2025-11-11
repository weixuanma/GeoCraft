import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureProjection(nn.Module):
    def __init__(self, cnn_backbone="ResNet50", num_cnn_layers=5, feature_dim=256, lambda_decay=0.8):
        super().__init__()
        self.num_cnn_layers = num_cnn_layers
        self.feature_dim = feature_dim
        self.lambda_decay = lambda_decay
        
        if cnn_backbone == "ResNet50":
            self.cnn = models.resnet50(pretrained=True)
            self.cnn_layers = self._extract_cnn_layers()
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        self.proj_matrices = nn.ModuleList([
            nn.Linear(layer.out_channels, feature_dim) for layer in self.cnn_layers
        ])

    def _extract_cnn_layers(self):
        layers = [
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
            self.cnn.layer1,
            self.cnn.layer2,
            self.cnn.layer3,
            self.cnn.layer4
        ]
        return layers[:self.num_cnn_layers]

    def forward(self, img, t, pc_coords, K, Rt):
        batch_size, _, img_h, img_w = img.shape
        num_points = pc_coords.shape[1]
        
        cnn_feats = []
        x = img
        for layer, proj_mat in zip(self.cnn_layers, self.proj_matrices):
            x = layer(x)
            x_proj = proj_mat(x.permute(0, 2, 3, 1)).permute(0, 3, 2, 1)
            cnn_feats.append(x_proj)
        
        t_norm = t / self.num_timesteps if hasattr(self, "num_timesteps") else 0.0
        w_t = torch.exp(-self.lambda_decay * t_norm).view(batch_size, 1, 1)
        
        proj_feats = []
        for feat in cnn_feats:
            feat_h, feat_w = feat.shape[2], feat.shape[3]
            scale_h = img_h / feat_h
            scale_w = img_w / feat_w
            
            pc_hom = torch.cat([pc_coords, torch.ones(batch_size, num_points, 1, device=pc_coords.device)], dim=-1)
            img_coords = torch.bmm(pc_hom, K.bmm(Rt).transpose(1, 2))
            img_coords = img_coords[..., :2] / img_coords[..., 2:3]
            
            img_coords[..., 0] /= (img_w - 1)
            img_coords[..., 1] /= (img_h - 1)
            img_coords = img_coords * 2 - 1
            
            feat_sampled = F.grid_sample(
                feat,
                img_coords.unsqueeze(1),
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            ).squeeze(1)
            proj_feats.append(feat_sampled)
        
        proj_feat = torch.stack(proj_feats, dim=2).mean(dim=2) * w_t
        return proj_feat