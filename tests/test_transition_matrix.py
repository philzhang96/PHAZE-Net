# tests/test_transition_matrix.py

from InERTIA.transition_matrix import EmotionTransitionMatrix

etm = EmotionTransitionMatrix()

print("✅ Transition Matrix Loaded\n")

for current in etm.get_emotions():
    next_emotion, prob = etm.get_next_emotion(current)
    vector = etm.get_probabilities(current)

    print(f"▶ From: {current}")
    print(f"   Next: {next_emotion} (p = {prob:.2f})")
    print("   Full Transition Probabilities:")
    for emo, prob in vector.items():
        print(f"      {emo}: {prob:.2f}")
    print()
