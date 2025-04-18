# InERTIA/transition_detector.py

import numpy as np
from .state_space import EmotionStateSpace

class TransitionDetector:
    def __init__(self, z_threshold=2.0):
        self.z_threshold = z_threshold
        self.state_space = EmotionStateSpace()

    def should_transition(self, current_emotion, momentum_va):
        # Support both tuple/list and numpy array
        if isinstance(momentum_va, (tuple, list, np.ndarray)) and len(momentum_va) == 2:
            v, a = float(momentum_va[0]), float(momentum_va[1])
        else:
            raise ValueError("Momentum vector must be a tuple, list, or array of (valence, arousal)")

        mean = self.state_space.get_mean(current_emotion)
        std = self.state_space.get_std(current_emotion)

        if mean is None or std is None:
            raise ValueError(f"Emotion '{current_emotion}' not found in EmotionStateSpace.")

        z_v = (v - mean[0]) / std[0]
        z_a = (a - mean[1]) / std[1]
        z_score = np.sqrt(z_v**2 + z_a**2)

        return z_score > self.z_threshold
