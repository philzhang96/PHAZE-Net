import torch
from models.emotion_predictor import EmotionPredictor

def test_emotion_predictor():
    # Simulate input: batch of fused features (e.g. from AGFW)
    batch_size = 4
    input_dim = 256  # Should match what AGFW outputs
    dummy_features = torch.randn(batch_size, input_dim)

    # Instantiate model
    model = EmotionPredictor(input_dim=input_dim, num_emotions=8)
    model.eval()  # We’re just testing

    # Run forward pass
    with torch.no_grad():
        logits, va = model(dummy_features)

    # Check output shapes
    assert logits.shape == (batch_size, 8), f"Expected logits shape (4, 8), got {logits.shape}"
    assert va.shape == (batch_size, 2), f"Expected VA shape (4, 2), got {va.shape}"

    print("✅ EmotionPredictor test passed.")
    print(f"   Logits shape: {logits.shape}")
    print(f"   VA shape:     {va.shape}")

if __name__ == "__main__":
    test_emotion_predictor()
