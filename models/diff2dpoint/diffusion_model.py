import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps=1000, alpha_init=1.0, alpha_final=1e-4, point_dim=3):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.point_dim = point_dim
        
        self.alphas = self._compute_alphas()
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.denoise_net = nn.Sequential(
            nn.Linear(point_dim + 256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, point_dim)
        )

    def _compute_alphas(self):
        alphas = torch.linspace(self.alpha_init, self.alpha_final, self.num_timesteps)
        return alphas

    def forward_diffusion(self, x0, t):
        batch_size = x0.shape[0]
        alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
        return xt, eps

    def reverse_denoise(self, xt, t, proj_feat):
        batch_size = xt.shape[0]
        t_emb = self._timestep_embedding(t, 128).view(batch_size, 1, 128)
        
        xt_flat = xt.view(batch_size, -1, self.point_dim)
        combined = torch.cat([xt_flat, proj_feat, t_emb.expand(-1, xt_flat.shape[1], -1)], dim=-1)
        
        eps_pred = self.denoise_net(combined)
        eps_pred = eps_pred.view_as(xt)
        
        alpha = self.alphas[t].view(batch_size, 1, 1)
        alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1)
        
        x_prev = (1 / torch.sqrt(alpha)) * (xt - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_pred)
        sigma_t = torch.sqrt((1 - alpha) * (1 - self.alpha_bars[t-1]) / (1 - alpha_bar)) if t[0] > 0 else 0.0
        if sigma_t > 0:
            x_prev += sigma_t * torch.randn_like(x_prev)
        return x_prev, eps_pred

    def _timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x0, t, proj_feat):
        xt, eps = self.forward_diffusion(x0, t)
        x_prev, eps_pred = self.reverse_denoise(xt, t, proj_feat)
        return x_prev, eps, eps_pred