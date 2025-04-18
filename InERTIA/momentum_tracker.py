# InERTIA/momentum_tracker.py

import numpy as np

class MomentumTracker:
    def __init__(self, alpha=0.6):
        """
        Initializes the momentum tracker.

        Args:
            alpha (float): Momentum weighting factor (0 < alpha ≤ 1).
                           Higher values make the system more reactive,
                           lower values make it smoother.
        """
        self.alpha = alpha
        self.momentum = None  # Initially undefined

    def reset(self):
        """Resets the internal momentum state."""
        self.momentum = None

    def update(self, valence, arousal):
        """
        Updates the internal momentum state given new VA input.

        Args:
            valence (float): New valence value.
            arousal (float): New arousal value.

        Returns:
            tuple: Updated momentum vector (valence, arousal)
        """
        current = np.array([valence, arousal], dtype=np.float32)

        if self.momentum is None:
            self.momentum = current
        else:
            self.momentum = self.alpha * current + (1 - self.alpha) * self.momentum

        return tuple(self.momentum)  # ✅ Ensures unpackable output

    def get_momentum(self):
        """Returns the current momentum vector."""
        return self.momentum
