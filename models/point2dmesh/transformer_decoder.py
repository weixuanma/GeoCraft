import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, hidden_dim=512, num_heads=8, dropout=0.1, point_feat_dim=256, vocab_size=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(512, hidden_dim)
        
        self.point_proj = nn.Linear(point_feat_dim, hidden_dim)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, mesh_tokens, point_feat):
        seq_len = mesh_tokens.shape[1]
        batch_size = mesh_tokens.shape[0]
        
        pos_ids = torch.arange(seq_len, device=mesh_tokens.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.embedding(mesh_tokens) + self.pos_embedding(pos_ids)
        
        point_feat_proj = self.point_proj(point_feat).unsqueeze(1).expand(-1, seq_len, -1)
        
        tgt_mask = self._generate_causal_mask(seq_len, device=mesh_tokens.device)
        out = self.decoder(
            tgt=token_emb,
            memory=point_feat_proj,
            tgt_mask=tgt_mask
        )
        logits = self.fc_out(out)
        return logits

    def _generate_causal_mask(self, seq_len, device):
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate(self, point_feat, seq_len=512, start_token=0, sampling_strategy="nucleus", nucleus_p=0.95):
        batch_size = point_feat.shape[0]
        device = point_feat.device
        
        tokens = torch.tensor([[start_token]] * batch_size, device=device)
        for _ in range(seq_len - 1):
            logits = self.forward(tokens, point_feat)[:, -1, :]
            if sampling_strategy == "nucleus":
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                keep = cumulative_probs <= nucleus_p
                keep[:, 0] = True
                sorted_probs = sorted_probs * keep
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices.gather(1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = logits.argmax(dim=-1).unsqueeze(1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens