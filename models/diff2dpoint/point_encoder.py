import torch
import torch.nn as nn
import torch.nn.functional as F

class PointEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pc):
        batch_size, num_points, _ = pc.shape
        
        local_feat = self.mlp1(pc)
        global_feat = local_feat.max(dim=1)[0].unsqueeze(1).expand(-1, num_points, -1)
        
        combined = torch.cat([local_feat, global_feat], dim=-1)
        final_feat = self.mlp2(combined)
        return final_feat