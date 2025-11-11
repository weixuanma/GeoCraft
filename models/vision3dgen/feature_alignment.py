import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.diff2dpoint.point_encoder import PointEncoder

class FeatureAlignment(nn.Module):
    def __init__(self, image_encoder="ViT-L/16", image_feat_dim=768, latent_dim=128*8, sparse_resnet_blocks=4):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.latent_dim = latent_dim
        
        if image_encoder == "ViT-L/16":
            self.image_encoder = models.vit_l_16(pretrained=True)
            self.image_proj = nn.Linear(image_feat_dim, latent_dim)
        else:
            raise ValueError(f"Unsupported image encoder: {image_encoder}")
        
        self.sparse_resnet = self._build_sparse_resnet(sparse_resnet_blocks, latent_dim)
        
        self.mesh_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.l2_loss = nn.MSELoss()

    def _build_sparse_resnet(self, num_blocks, dim):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Conv3d(dim, dim, 3, padding=1),
                nn.BatchNorm3d(dim),
                nn.ReLU(),
                nn.Conv3d(dim, dim, 3, padding=1),
                nn.BatchNorm3d(dim)
            ))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def align_image(self, img, latent_triplane):
        batch_size = img.shape[0]
        img_feat = self.image_encoder(img).last_hidden_state[:, 1:, :]
        img_feat_proj = self.image_proj(img_feat)
        
        latent_flat = latent_triplane.view(batch_size, -1, self.latent_dim)
        img_feat_flat = img_feat_proj.view(batch_size, -1, self.latent_dim)
        
        img_mean = img_feat_flat.mean(dim=1)
        img_var = img_feat_flat.var(dim=1)
        latent_mean = latent_flat.mean(dim=1)
        latent_var = latent_flat.var(dim=1)
        
        kl = self.kl_loss(
            F.log_softmax(img_feat_flat, dim=-1),
            F.softmax(latent_flat, dim=-1)
        )
        return img_feat_proj, kl

    def align_mesh(self, voxel_mesh, point_feat):
        batch_size = voxel_mesh.shape[0]
        mesh_feat = self.sparse_resnet(voxel_mesh)
        mesh_feat_flat = mesh_feat.view(batch_size, -1, self.latent_dim)
        
        point_feat_proj = nn.Linear(point_feat.shape[-1], self.latent_dim)(point_feat)
        point_feat_flat = point_feat_proj.view(batch_size, -1, self.latent_dim)
        
        attn_out, _ = self.mesh_attn(mesh_feat_flat, point_feat_flat, point_feat_flat)
        
        sample_idx = torch.randint(point_feat_flat.shape[1], (mesh_feat_flat.shape[1],), device=mesh_feat_flat.device)
        point_feat_sampled = point_feat_flat.gather(1, sample_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.latent_dim))
        
        l2 = self.l2_loss(attn_out, point_feat_sampled)
        return attn_out, l2

    def forward(self, img, voxel_mesh, point_feat, latent_triplane):
        img_feat_aligned, kl_loss = self.align_image(img, latent_triplane)
        mesh_feat_aligned, l2_loss = self.align_mesh(voxel_mesh, point_feat)
        total_loss = kl_loss + l2_loss
        return img_feat_aligned, mesh_feat_aligned, total_loss