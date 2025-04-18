from InERTIA.inertia_controller import InERTIAController

# Simulated valence-arousal inputs across time
test_sequence = [
    (0.75, 0.60),   # Strong happiness
    (0.50, 0.30),   # Fading
    (0.30, 0.00),   # Approaching neutral/low valence
    (0.10, -0.30),  # Drifting toward sadness
    (-0.50, -0.50), # Likely triggers transition
]

controller = InERTIAController(momentum_alpha=0.8, transition_lambda=2.0)
print("🧪 InERTIA Controller Runtime:")

for i, va in enumerate(test_sequence):
    emotion, changed = controller.step(va)
    status = "🔁 Transitioned" if changed else "✅ Stayed"
    print(f"Frame {i+1}: V={va[0]:.2f}, A={va[1]:.2f} → Emotion: {emotion:10s} | {status}")
