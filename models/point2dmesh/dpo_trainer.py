import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOTrainer(nn.Module):
    def __init__(self, model, temperature=0.1):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, point_feat, mesh_tokens_pos, mesh_tokens_neg):
        logits_pos = self.model(mesh_tokens_pos[:, :-1], point_feat)
        logits_neg = self.model(mesh_tokens_neg[:, :-1], point_feat)
        
        log_probs_pos = F.log_softmax(logits_pos, dim=-1).gather(2, mesh_tokens_pos[:, 1:].unsqueeze(2)).squeeze(2).sum(dim=1)
        log_probs_neg = F.log_softmax(logits_neg, dim=-1).gather(2, mesh_tokens_neg[:, 1:].unsqueeze(2)).squeeze(2).sum(dim=1)
        
        log_ratio = (log_probs_pos - log_probs_neg) / self.temperature
        labels = torch.ones_like(log_ratio)
        loss = self.loss_fn(log_ratio, labels)
        return loss

    def fine_tune(self, dataloader, optimizer, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                point_feat, pos_tokens, neg_tokens = batch
                optimizer.zero_grad()
                loss = self.forward(point_feat, pos_tokens, neg_tokens)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Fine-tune Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")