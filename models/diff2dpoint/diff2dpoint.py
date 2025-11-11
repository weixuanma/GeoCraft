import torch
import torch.nn as nn
from .diffusion_model import DiffusionModel
from .feature_projection import FeatureProjection
from .point_encoder import PointEncoder

class Diff2DPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.point_encoder = PointEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=config["feature_projection"]["feature_dim"]
        )
        
        self.feature_projection = FeatureProjection(
            cnn_backbone=config["feature_projection"]["cnn_backbone"],
            num_cnn_layers=config["feature_projection"]["num_cnn_layers"],
            feature_dim=config["feature_projection"]["feature_dim"],
            lambda_decay=config["feature_projection"]["lambda_decay"]
        )
        self.feature_projection.num_timesteps = config["diffusion"]["num_timesteps"]
        
        self.diffusion_model = DiffusionModel(
            num_timesteps=config["diffusion"]["num_timesteps"],
            alpha_init=config["diffusion"]["alpha_init"],
            alpha_final=config["diffusion"]["alpha_final"],
            point_dim=3
        )

    def forward(self, img, pc_gt, K, Rt):
        batch_size = img.shape[0]
        t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=img.device)
        
        pc_feat = self.point_encoder(pc_gt)
        proj_feat = self.feature_projection(img, t, pc_gt, K, Rt)
        
        x_prev, eps, eps_pred = self.diffusion_model(pc_gt, t, proj_feat)
        return x_prev, eps, eps_pred

    def generate_point_cloud(self, img, K, Rt, num_points=2048):
        self.eval()
        batch_size = img.shape[0]
        device = img.device
        
        xt = torch.randn(batch_size, num_points, 3, device=device)
        for t in range(self.diffusion_model.num_timesteps - 1, -1, -1):
            t_tensor = torch.tensor([t] * batch_size, device=device)
            proj_feat = self.feature_projection(img, t_tensor, xt, K, Rt)
            xt, _ = self.diffusion_model.reverse_denoise(xt, t_tensor, proj_feat)
        return xt