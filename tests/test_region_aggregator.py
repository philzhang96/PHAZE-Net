import torch
from InERTIA.region_aggregator import RegionAggregator

print("🧪 Emotion-Specific Region Aggregator Test")

# Simulated input
region_outputs = {
    "Mouth::Upper lip (upper)": torch.randn(128),
    "Eyes::Right upper eyelid": torch.randn(128),
    "Eyes::Left upper eyelid": torch.randn(128),
    "Nose::Nose Bridge": torch.randn(128),
    "Cheeks::Left cheekbone": torch.randn(128)
}

confidences = {
    "Mouth::Upper lip (upper)": 0.8,
    "Eyes::Right upper eyelid": 0.6,
    "Eyes::Left upper eyelid": 0.5,
    "Nose::Nose Bridge": 0.7,
    "Cheeks::Left cheekbone": 0.9
}

# Emotion to test with
emotion = "Happiness"

# Run aggregation
aggregator = RegionAggregator()
aggregated_vector = aggregator.aggregate(region_outputs, confidences, emotion)

# Print results
print(f"🎯 Using emotion: {emotion}")
print("✅ Aggregated Output Shape:", aggregated_vector.shape)
print("🔍 Aggregated Vector (first 5 values):", aggregated_vector[:5])
