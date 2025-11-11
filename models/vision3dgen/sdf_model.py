import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import marching_cubes

class SDFModel(nn.Module):
    def __init__(self, mlp_layers=8, mlp_hidden_dims=[256, 512, 512, 1024, 1024, 512, 512, 256], output_dim=1, activation="ReLU"):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.output_dim = output_dim
        
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.mlp = self._build_mlp(mlp_hidden_dims)

    def _build_mlp(self, hidden_dims):
        layers = []
        input_dim = 3 + (128*8)*3
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation)
            input_dim = dim
        layers.append(nn.Linear(input_dim, self.output_dim))
        return nn.Sequential(*layers)

    def _sample_triplane_feat(self, triplane, coords):
        batch_size = coords.shape[0]
        num_points = coords.shape[1]
        device = coords.device
        
        coords_norm = (coords + 1) / 2
        coords_norm = coords_norm.clamp(0, 1)
        
        xy_feat = F.grid_sample(
            triplane[:, 0],
            coords_norm[..., [0, 1]].unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze(1)
        
        yz_feat = F.grid_sample(
            triplane[:, 1],
            coords_norm[..., [1, 2]].unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze(1)
        
        xz_feat = F.grid_sample(
            triplane[:, 2],
            coords_norm[..., [0, 2]].unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze(1)
        
        combined_feat = torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)
        return combined_feat

    def forward(self, triplane, coords):
        triplane_feat = self._sample_triplane_feat(triplane, coords)
        coords_feat = self._pos_embedding(coords)
        combined = torch.cat([coords_feat, triplane_feat], dim=-1)
        sdf = self.mlp(combined)
        return sdf

    def _pos_embedding(self, coords):
        dim = 128
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0, device=coords.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=coords.device) * -emb)
        emb = coords.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def extract_surface(self, triplane, resolution=256, threshold=0.0):
        self.eval()
        with torch.no_grad():
            x = torch.linspace(-1, 1, resolution, device=triplane.device)
            y = torch.linspace(-1, 1, resolution, device=triplane.device)
            z = torch.linspace(-1, 1, resolution, device=triplane.device)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
            coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 3)
            
            sdf = self.forward(triplane, coords).view(resolution, resolution, resolution)
            sdf_np = sdf.cpu().numpy()
            
            verts, faces, normals, values = marching_cubes(sdf_np, level=threshold)
            return verts, faces, normals