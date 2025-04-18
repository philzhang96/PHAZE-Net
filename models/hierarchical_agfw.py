import torch
import torch.nn as nn

class AGFW(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AGFW, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, region_features):
        # region_features: (B, R, F)
        scores = self.attn(region_features)  # (B, R, 1)
        weights = torch.softmax(scores, dim=1)  # (B, R, 1)
        weighted_sum = (weights * region_features).sum(dim=1)  # (B, F)
        return weighted_sum, weights


class HierarchicalAGFW(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(HierarchicalAGFW, self).__init__()
        self.agfw_short = AGFW(input_dim, hidden_dim)
        self.agfw_medium = AGFW(input_dim, hidden_dim)
        self.agfw_long = AGFW(input_dim, hidden_dim)

        self.final_fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.ReLU()
        )

    def forward(self, short_feats, medium_feats, long_feats):
        # Each input: (B, R, F)
        short_out, w_short = self.agfw_short(short_feats)
        medium_out, w_medium = self.agfw_medium(medium_feats)
        long_out, w_long = self.agfw_long(long_feats)

        combined = torch.cat([short_out, medium_out, long_out], dim=-1)  # (B, 3F)
        fused = self.final_fusion(combined)  # (B, F)

        return fused, {
            "short_weights": w_short,
            "medium_weights": w_medium,
            "long_weights": w_long
        }
