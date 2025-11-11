import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diff2dpoint.point_encoder import PointEncoder

class TriplaneConstructor(nn.Module):
    def __init__(self, resolution=256, num_channels=128, num_center_points=1024, k_neighbors=32, sigma=0.1):
        super().__init__()
        self.resolution = resolution
        self.num_channels = num_channels
        self.num_center_points = num_center_points
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        
        self.point_encoder = PointEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=256
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.cross_plane_attn = nn.MultiheadAttention(
            embed_dim=num_channels * (resolution // 8) ** 2,
            num_heads=8,
            batch_first=True
        )
        
        self.down_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels*2, num_channels*4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels*4, num_channels*8, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def _density_weighted_sampling(self, pc):
        batch_size, num_points, _ = pc.shape
        device = pc.device
        
        dist_matrix = torch.cdist(pc, pc)
        density = torch.exp(-2 * self.sigma ** 2 / (dist_matrix[:, :, :self.k_neighbors] ** 2 + 1e-6)).mean(dim=2)
        
        center_points = []
        for b in range(batch_size):
            pc_b = pc[b]
            density_b = density[b]
            centers = []
            
            first_idx = density_b.argmax()
            centers.append(pc_b[first_idx])
            remaining = torch.ones(num_points, device=device, dtype=torch.bool)
            remaining[first_idx] = False
            
            for _ in range(self.num_center_points - 1):
                if not remaining.any():
                    break
                dist_to_centers = torch.cdist(pc_b[remaining], torch.stack(centers))
                min_dist = dist_to_centers.min(dim=1)[0]
                next_idx = remaining.nonzero()[min_dist.argmax()][0]
                centers.append(pc_b[next_idx])
                remaining[next_idx] = False
            
            while len(centers) < self.num_center_points:
                centers.append(pc_b[torch.randint(num_points, (1,)).item()])
            center_points.append(torch.stack(centers))
        return torch.stack(center_points)

    def _pos_embedding(self, coords):
        dim = coords.shape[-1]
        half_dim = 128 // 2
        emb = torch.log(torch.tensor(10000.0, device=coords.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=coords.device) * -emb)
        emb = coords.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, pc):
        batch_size = pc.shape[0]
        device = pc.device
        
        center_pc = self._density_weighted_sampling(pc)
        center_feat = self.point_encoder(center_pc)
        
        pos_emb = self._pos_embedding(center_pc)
        combined_feat = torch.cat([center_feat, pos_emb], dim=-1)
        global_feat = self.transformer_encoder(combined_feat)
        
        triplane = torch.zeros(batch_size, 3, self.num_channels, self.resolution, self.resolution, device=device)
        L = pc.max() - pc.min()
        
        for b in range(batch_size):
            centers_b = center_pc[b]
            feat_b = global_feat[b]
            
            xy_coords = ((centers_b[:, :2] + L/2) / L * self.resolution).long()
            xy_coords = xy_coords.clamp(0, self.resolution - 1)
            triplane[b, 0, :, xy_coords[:, 1], xy_coords[:, 0]] += feat_b.T
            
            yz_coords = ((centers_b[:, 1:] + L/2) / L * self.resolution).long()
            yz_coords = yz_coords.clamp(0, self.resolution - 1)
            triplane[b, 1, :, yz_coords[:, 1], yz_coords[:, 0]] += feat_b.T
            
            xz_coords = ((centers_b[:, [0, 2]] + L/2) / L * self.resolution).long()
            xz_coords = xz_coords.clamp(0, self.resolution - 1)
            triplane[b, 2, :, xz_coords[:, 1], xz_coords[:, 0]] += feat_b.T
        
        triplane_down = self.down_conv(triplane.view(-1, self.num_channels, self.resolution, self.resolution))
        triplane_flat = triplane_down.view(batch_size, 3, -1)
        attn_out, _ = self.cross_plane_attn(triplane_flat, triplane_flat, triplane_flat)
        
        latent_triplane = attn_out.view(batch_size, 3, self.num_channels*8, self.resolution//8, self.resolution//8)
        return latent_triplane