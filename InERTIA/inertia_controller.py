from InERTIA.momentum_tracker import MomentumTracker
from InERTIA.transition_detector import TransitionDetector
from InERTIA.transition_matrix import EmotionTransitionMatrix

class InERTIAController:
    def __init__(self, momentum_alpha=0.6, transition_lambda=2.0):
        self.momentum_tracker = MomentumTracker(alpha=momentum_alpha)
        self.transition_detector = TransitionDetector(z_threshold=transition_lambda)
        self.transition_matrix = EmotionTransitionMatrix()
        self.current_emotion = "Happiness"  # Set to a valid emotion in the matrix
    


    def reset(self, initial_emotion="Happiness"):
        """Resets internal state."""
        self.momentum_tracker.reset()
        self.current_emotion = initial_emotion

    
    def step(self, valence, arousal):
        """
        The step function takes in valence and arousal values,
        updates the momentum, checks if a transition occurs,
        and returns the updated emotion and whether it changed.
        """
        # Update the momentum tracker
        momentum = self.momentum_tracker.update(valence, arousal)

        # Use transition detector to check if the emotion should transition
        should_transition = self.transition_detector.should_transition(self.current_emotion, momentum)

        if should_transition:
            # Predict next emotion from the transition model
            next_emotion, prob = self.transition_matrix.get_next_emotion(self.current_emotion)
            self.current_emotion = next_emotion
            return self.current_emotion, True  # Emotion has changed
        else:
            return self.current_emotion, False  # Emotion stayed the same
    
   