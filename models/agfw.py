import torch
import torch.nn as nn
import torch.nn.functional as F

class AGFW(nn.Module):
    def __init__(self, num_regions, feature_dim):
        super(AGFW, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)
        self.num_regions = num_regions

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, chunk_size, num_regions, feature_dim]
        
        Returns:
            Tensor of shape [batch_size, chunk_size, feature_dim]
            (weighted average over ROIs per frame)
        """
        batch_size, chunk_size, num_regions, feature_dim = x.shape

        # Flatten all regions to compute attention
        x_reshaped = x.view(-1, feature_dim)  # [B * C * R, F]
        attn_scores = self.attention_fc(x_reshaped)  # [B * C * R, 1]
        attn_scores = attn_scores.view(batch_size, chunk_size, num_regions)  # [B, C, R]

        # Softmax over regions (last dim)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, C, R]

        # Multiply attention weights by region features
        weighted_sum = torch.sum(attn_weights.unsqueeze(-1) * x, dim=2)  # [B, C, F]

        return weighted_sum
