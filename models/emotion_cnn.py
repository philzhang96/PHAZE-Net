import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_emotions=8, feature_dim=256):
        super(EmotionCNN, self).__init__()

        # Backbone: Custom lightweight CNN for spatial feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 128, 1, 1]
        )

        # Flatten + feature vector
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, feature_dim)  # Flexible output dimension

        # Output heads
        self.classifier = nn.Linear(feature_dim, num_emotions)
        self.va_regressor = nn.Linear(feature_dim, 2)

    def forward(self, x):
        """
        Full model output: logits + valence/arousal regression.
        """
        x = self.backbone(x)
        x = self.flatten(x)
        features = self.fc(x)
        logits = self.classifier(features)
        va = self.va_regressor(features)
        return logits, va

    def extract_features(self, x):
        """
        Feature extraction mode: outputs intermediate vector after backbone and fc.
        Used for feeding into AGFW or InERTIA.
        """
        x = self.backbone(x)
        x = self.flatten(x)
        features = self.fc(x)
        return features
