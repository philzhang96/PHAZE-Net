import numpy as np
from preprocessing.temporal_chunker import multi_resolution_chunking

def generate_fake_sequence(length=60):
    """
    Generates a fake list of grayscale ROIs as NumPy arrays.
    """
    return [np.full((50, 50), fill_value=i, dtype=np.uint8) for i in range(length)]

def test_chunking():
    print("🧪 Testing temporal chunking...")

    # Simulate cleaned ROI output: 60 frames per region
    roi_dict = {
        "Eyes::Left": generate_fake_sequence(60),
        "Mouth::Upper lip (upper)": generate_fake_sequence(60)
    }

    # Config based on 3 FPS → short = 15f, medium = 30f, long = 45f
    chunk_config = {
        "short":  {"size": 15, "stride": 3},
        "medium": {"size": 30, "stride": 6},
        "long":   {"size": 45, "stride": 9}
    }

    chunked = multi_resolution_chunking(roi_dict, chunk_config)

    # Print structure and count
    for region, scales in chunked.items():
        print(f"📍 {region}")
        for scale, chunks in scales.items():
            print(f"   • {scale}: {len(chunks)} chunks of {len(chunks[0])} frames")

    print("✅ Temporal chunking test complete.")

if __name__ == "__main__":
    test_chunking()
