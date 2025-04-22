# InERTIA/smoother.py

class Smoother:
    def __init__(self, alpha=0.6):
        """
        Implements exponential smoothing for valence-arousal values.
        Args:
            alpha (float): Smoothing factor. Higher = more responsive to new input.
        """
        self.alpha = alpha
        self.smoothed = None

    def reset(self):
        """Resets the smoother to its initial state."""
        self.smoothed = None

    def update(self, valence, arousal):
        """
        Updates the smoothed valence-arousal estimate.
        Args:
            valence (float): Current valence input.
            arousal (float): Current arousal input.
        Returns:
            Tuple of (smoothed_valence, smoothed_arousal)
        """
        current = (valence, arousal)

        if self.smoothed is None:
            self.smoothed = current
        else:
            sv, sa = self.smoothed
            cv, ca = current
            new_sv = self.alpha * cv + (1 - self.alpha) * sv
            new_sa = self.alpha * ca + (1 - self.alpha) * sa
            self.smoothed = (new_sv, new_sa)

        return self.smoothed
