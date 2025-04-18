import numpy as np
from InERTIA.momentum_tracker import MomentumTracker

# Create tracker with known alpha
alpha = 0.6
tracker = MomentumTracker(alpha=alpha)

# Example frame-level valence/arousal predictions
va_sequence = [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.6, 0.6),
    (0.4, 0.2),
]

print("Testing MomentumTracker...\n")
for t, (v, a) in enumerate(va_sequence):
    momentum = tracker.update(v, a)
    print(f"Frame {t + 1}:")
    print(f" - Input VA:      ({v:.2f}, {a:.2f})")
    print(f" - Momentum VA:   ({momentum[0]:.4f}, {momentum[1]:.4f})\n")

# Optional: reset test
tracker.reset()
print("✅ Reset successful. Current momentum:", tracker.get_momentum())
