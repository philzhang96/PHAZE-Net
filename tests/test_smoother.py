# tests/test_smoother.py

from InERTIA.smoother import Smoother

print("🧪 Testing Smoother")

# Create smoother with exponential smoothing (alpha=0.6)
smoother = Smoother(alpha=0.6)

# Simulated valence-arousal predictions (raw)
va_sequence = [
    (0.1, 0.1),
    (0.15, 0.12),
    (0.9, 0.85),
    (0.16, 0.11),
    (0.12, 0.09),
    (0.14, 0.10)
]

for i, (v, a) in enumerate(va_sequence, 1):
    smoothed = smoother.update(v, a)
    print(f"Frame {i}: Raw=({v:.2f}, {a:.2f}) | Smoothed=({smoothed[0]:.2f}, {smoothed[1]:.2f})")
