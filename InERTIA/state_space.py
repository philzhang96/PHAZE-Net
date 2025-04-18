# InERTIA/state_space.py

import numpy as np

class EmotionStateSpace:
    def __init__(self):
        # Define emotion list (ℇ set)
        self.emotions = [
            "Anger", "Contempt", "Disgust", "Fear",
            "Happiness", "Sadness", "Surprise", "Neutral"
        ]

        # Updated mean valence-arousal coordinates per emotion (μ_v, μ_a)
        self.means = {
            "Anger":     (-0.474650, 0.575124),
            "Contempt":  (-0.512445, 0.579456),
            "Disgust":   (-0.659354, 0.417783),
            "Fear":      (-0.294369, 0.708742),
            "Happiness": ( 0.739087, 0.292775),
            "Sadness":   (-0.593840, 0.159666),
            "Surprise":  ( 0.333774, 0.703797),
            "Neutral":   ( 0.008214, 0.015013),
        }

        # Updated standard deviations (σ_v, σ_a)
        self.stds = {
            "Anger":     (0.184825, 0.266441),
            "Contempt":  (0.189570, 0.122281),
            "Disgust":   (0.171846, 0.208117),
            "Fear":      (0.114288, 0.205479),
            "Happiness": (0.187694, 0.209043),
            "Sadness":   (0.208397, 0.161865),
            "Surprise":  (0.202275, 0.199077),
            "Neutral":   (0.118120, 0.091148),
        }

    def get_emotion_list(self):
        return self.emotions

    def get_mean(self, emotion):
        return self.means.get(emotion)

    def get_std(self, emotion):
        return self.stds.get(emotion)

    def get_all_means_stds(self):
        """Returns a dict of emotion: (mean, std) pairs."""
        return {
            emotion: (self.means[emotion], self.stds[emotion])
            for emotion in self.emotions
        }

# Optional test
if __name__ == "__main__":
    ess = EmotionStateSpace()
    for emotion in ess.get_emotion_list():
        mean = ess.get_mean(emotion)
        std = ess.get_std(emotion)
        print(f"{emotion}: μ={mean}, σ={std}")
