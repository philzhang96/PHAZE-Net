# inertia/region_weights.py

REGION_WEIGHTS = {
    "Happiness": {
        "Cheeks": 0.3,
        "Mouth": 0.4,
        "Eyes": 0.3
    },
    "Sadness": {
        "Eyebrows": 0.4,
        "Chin": 0.2,
        "Lips": 0.4  # Mapped to "Mouth" in sub-feature aggregation
    },
    "Surprise": {
        "Eyebrows": 0.3,
        "Eyes": 0.4,
        "Mouth": 0.3
    },
    "Fear": {
        "Eyebrows": 0.35,
        "Eyes": 0.35,
        "Mouth": 0.3
    },
    "Anger": {
        "Eyebrows": 0.3,
        "Eyes": 0.3,
        "Mouth": 0.3,
        "Nose": 0.1
    },
    "Disgust": {
        "Nose": 0.4,
        "Mouth": 0.3,
        "Chin": 0.3
    },
    "Contempt": {
        "Mouth": 1.0
    }
}
