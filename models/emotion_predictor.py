import torch
import torch.nn as nn

class EmotionPredictor(nn.Module):
    """
    Predicts discrete emotion class and continuous valence/arousal values from fused features.
    Designed to sit after AGFW (or any temporal fusion module).
    """
    def __init__(self, input_dim=256, num_emotions=8):
        super(EmotionPredictor, self).__init__()
        
        self.dropout = nn.Dropout(0.3)
        
        # Shared MLP layer before branching
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Output head: emotion classification (logits for softmax)
        self.classifier = nn.Linear(128, num_emotions)

        # Output head: valence/arousal regression
        self.va_regressor = nn.Linear(128, 2)

    def forward(self, x):
        """
        x: fused feature vector (e.g., from AGFW) with shape [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_emotions]
            va:     [batch_size, 2]  # valence and arousal
        """
        x = self.dropout(x)
        features = self.shared_fc(x)
        logits = self.classifier(features)
        va = self.va_regressor(features)
        return logits, va
