# tests/test_transition_detector.py

from InERTIA.transition_detector import TransitionDetector

# Set up the detector with default Z-threshold
detector = TransitionDetector(z_threshold=2.0)
current_emotion = "Happiness"

# Example momentum inputs to test
test_momentums = [
    (0.75, 0.60),   # Perfectly aligned with Happiness
    (0.50, 0.30),   # Slightly off
    (0.30, 0.00),   # More divergent
    (0.10, -0.30),  # Strongly deviates
    (-0.50, -0.50), # Very far
]

print(f"🧪 Testing transition trigger from emotion: {current_emotion}\n")

for i, momentum_va in enumerate(test_momentums, 1):
    # ✅ Correct
    result = detector.should_transition(current_emotion, momentum_va)
    v_str = f"{momentum_va[0]:.2f}"
    a_str = f"{momentum_va[1]:.2f}"
    status = "🔁 Triggered" if result else "✅ Remain"
    print(f"Frame {i}: Momentum (V={v_str}, A={a_str}) → {status}")
