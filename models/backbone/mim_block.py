import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidSparseAttention(nn.Module):
    """Pyramid Sparse Attention: Coarse-to-fine token selection"""
    def __init__(self, dim, num_heads=4, sparse_ratio=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparse_ratio = sparse_ratio
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # [B, num_tokens, dim]
        num_sparse = int(N * self.sparse_ratio)
        
        # Project to Q, K, V
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, heads, head_dim, N]
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        
        # Calculate attention scores & select sparse tokens
        attn_scores = torch.matmul(Q, K) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))  # [B, heads, N, N]
        attn_values = torch.mean(attn_scores, dim=2)  # [B, heads, N]
        _, top_indices = torch.topk(attn_values, num_sparse, dim=-1)  # [B, heads, num_sparse]
        
        # Gather sparse tokens for V
        sparse_V = []
        for b in range(B):
            for h in range(self.num_heads):
                indices = top_indices[b, h].unsqueeze(1).expand(-1, self.head_dim)  # [num_sparse, head_dim]
                v = V[b, h].gather(0, indices)  # [num_sparse, head_dim]
                sparse_V.append(v.unsqueeze(0).unsqueeze(0))  # [1, 1, num_sparse, head_dim]
        sparse_V = torch.cat(sparse_V, dim=0).reshape(B, self.num_heads, num_sparse, self.head_dim)  # [B, heads, num_sparse, head_dim]
        
        # Sparse attention calculation
        sparse_attn_scores = attn_scores.gather(3, top_indices.unsqueeze(2).expand(-1, -1, N, -1))  # [B, heads, N, num_sparse]
        sparse_attn_weights = F.softmax(sparse_attn_scores, dim=-1)
        sparse_out = torch.matmul(sparse_attn_weights, sparse_V)  # [B, heads, N, head_dim]
        
        # Reshape to output
        out = sparse_out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out

class SPTModule(nn.Module):
    """Sparse Pyramid Transformer Module: Multi-scale attention fusion with coarse-to-fine tokens"""
    def __init__(self, in_dim, token_levels=3, num_heads=4, sparse_ratio=0.6, pyramid_scales=[4, 2, 1]):
        super().__init__()
        self.token_levels = token_levels
        self.pyramid_scales = pyramid_scales
        self.dim = in_dim
        
        # Token projection for each scale
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, 1) for _ in pyramid_scales
        ])
        
        # Pyramid Sparse Attention (PSA) for each level
        self.psa_layers = nn.ModuleList([
            PyramidSparseAttention(in_dim, num_heads, sparse_ratio) for _ in range(token_levels)
        ])
        
        # Cross-scale fusion
        self.cross_fusion = nn.Conv2d(in_dim * len(pyramid_scales), in_dim, 1)
        self.batch_norm = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        scale_feats = []
        
        # Step 1: Extract pyramid scale features
        for idx, (scale, proj) in enumerate(zip(self.pyramid_scales, self.scale_projs)):
            # Resize to scale
            feat = F.interpolate(
                x, 
                size=(H//scale, W//scale), 
                mode='bilinear', 
                align_corners=True
            )
            # Project to target dim
            feat = proj(feat)
            # Reshape to token sequence
            feat = feat.permute(0, 2, 3, 1).reshape(B, -1, self.dim)  # [B, N_tokens, dim]
            # Apply PSA (coarse-to-fine token selection)
            for psa in self.psa_layers:
                feat = psa(feat)
            # Reshape back to 2D
            feat = feat.reshape(B, H//scale, W//scale, self.dim).permute(0, 3, 1, 2)  # [B, dim, H/scale, W/scale]
            # Resize back to original size
            feat = F.interpolate(
                feat, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            )
            scale_feats.append(feat)
        
        # Step 2: Cross-scale feature fusion
        fused_feat = torch.cat(scale_feats, dim=1)  # [B, dim*scales, H, W]
        fused_feat = self.cross_fusion(fused_feat)
        fused_feat = self.batch_norm(fused_feat)
        
        return fused_feat