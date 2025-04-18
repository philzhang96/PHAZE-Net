# InERTIA/transition_matrix.py

import numpy as np
import random

class EmotionTransitionMatrix:
    def __init__(self):
        self.emotions = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

        # Final pre-averaged transition matrix, with self-transitions set to 0
        self.matrix = np.array([
            # To:        A     C     D     F     H     S     Su
            [ 0.00, 0.05, 0.13, 0.12, 0.07, 0.52, 0.11],  # From Anger
            [ 0.09, 0.00, 0.14, 0.07, 0.10, 0.45, 0.15],  # From Contempt
            [ 0.10, 0.13, 0.00, 0.14, 0.07, 0.48, 0.08],  # From Disgust
            [ 0.08, 0.09, 0.10, 0.00, 0.05, 0.56, 0.12],  # From Fear
            [ 0.07, 0.05, 0.06, 0.06, 0.00, 0.61, 0.15],  # From Happiness
            [ 0.13, 0.08, 0.10, 0.14, 0.10, 0.00, 0.45],  # From Sadness
            [ 0.09, 0.06, 0.08, 0.09, 0.13, 0.41, 0.00],  # From Surprise
        ])

        # Normalise rows so they sum to 1 (ignoring self-transitions)
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix = self.matrix / row_sums

    def get_emotions(self):
        return self.emotions

    def get_next_emotion(self, current_emotion):
        """
        Given the current emotion, sample the next emotion from the row's distribution.
        Returns a tuple of (emotion, probability).
        """
        if current_emotion not in self.emotions:
            raise ValueError(f"Unknown emotion: {current_emotion}")
        row = self.matrix[self.emotions.index(current_emotion)]
        next_emotion = random.choices(self.emotions, weights=row)[0]
        prob = row[self.emotions.index(next_emotion)]
        return next_emotion, prob


    def get_probabilities(self, current_emotion):
        """
        Return the full transition probability vector from the given emotion.
        """
        if current_emotion not in self.emotions:
            raise ValueError(f"Emotion '{current_emotion}' not in defined emotion list.")

        idx = self.emotions.index(current_emotion)
        return dict(zip(self.emotions, self.matrix[idx]))
