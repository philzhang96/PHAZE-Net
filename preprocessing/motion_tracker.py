import numpy as np

# Import region landmark index map (you can import this from roi_extractor if you modularise it)
REGION_LANDMARKS = {
    "Cheeks": {
        "Left cheekbone": [50],
        "Right Cheekbone": [280],
    },
    "Mouth": {
        "Upper lip (upper)": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "Upper lip (lower)": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        "Lower lip (upper)": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        "Lower lip (lower)": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "Left mouth corner (looking at face)": [61],
        "Right mouth corner (looking at face)": [291],
    },
    "Eyes": {
        "Right upper eyelid": [362, 398, 384, 385, 386, 387, 388, 466, 263],
        "Right lower eyelid": [362, 382, 381, 380, 374, 373, 390, 249, 263],
        "Left upper eyelid": [133, 173, 157, 158, 159, 160, 161, 246, 33],
        "Left lower eyelid": [133, 155, 154, 153, 145, 144, 163, 7, 33],
        "Right outer eye corner": [446],
        "Right inner eye corner": [463],
        "Left outer eye corner": [226],
        "Left inner eye corner": [243],
    },
    "Eyes brows": {
        "Left eyebrow (upper)": [107, 66, 105, 63, 70],
        "Right eyebrow (upper)": [336, 296, 334, 293, 300],
    },
    "Chin": {
        "Chip Tip": [152],
    },
    "Nose": {
        "Nose Bridge": [6, 197, 195, 5],
    },
}

def get_region_centroid(landmarks, indices):
    valid_points = [landmarks[i] for i in indices if not np.isnan(landmarks[i][0])]
    if not valid_points:
        return None
    return np.mean(valid_points, axis=0)  # (x, y)

def compute_motion_from_landmarks(landmark_sequence):
    """
    Computes Euclidean movement (frame-to-frame) of each subregion in Ulrika's ROI map.

    Args:
        landmark_sequence (np.ndarray): shape (T, 468, 2)

    Returns:
        motion_dict: dict of {region_name: motion_array} where shape = (T-1,)
    """
    T = landmark_sequence.shape[0]
    motion_dict = {}

    for primary_region, subfeatures in REGION_LANDMARKS.items():
        for sub_name, indices in subfeatures.items():
            region_id = f"{primary_region}::{sub_name}"
            motion_array = []

            for t in range(1, T):
                prev_landmarks = landmark_sequence[t - 1]
                curr_landmarks = landmark_sequence[t]

                prev_center = get_region_centroid(prev_landmarks, indices)
                curr_center = get_region_centroid(curr_landmarks, indices)

                if prev_center is None or curr_center is None:
                    motion_array.append(0.0)  # No movement recorded
                else:
                    motion = np.linalg.norm(curr_center - prev_center)
                    motion_array.append(motion)

            motion_dict[region_id] = np.array(motion_array)

    return motion_dict
