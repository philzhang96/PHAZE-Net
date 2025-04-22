import cv2
import numpy as np

# 👇 Paste the full REGION_LANDMARKS dictionary here (truncated for preview)
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

def extract_rois_from_landmarks(image, landmarks, margin=10):
    """
    Extract ROIs from an image using MediaPipe landmark points and Ulrika's mapping.
    """
    h, w, _ = image.shape
    rois = {}

    for primary_region, subfeatures in REGION_LANDMARKS.items():
        for sub_name, indices in subfeatures.items():
            points = [landmarks[i] for i in indices if i < len(landmarks)]
            if not points:
                continue
            xs, ys = zip(*points)
            x_min = max(min(xs) - margin, 0)
            x_max = min(max(xs) + margin, w)
            y_min = max(min(ys) - margin, 0)
            y_max = min(max(ys) + margin, h)
            crop = image[y_min:y_max, x_min:x_max]
            rois[f"{primary_region}::{sub_name}"] = crop

    return rois

def fallback_rois(image):
    """
    Use fallback regions (hardcoded coordinates) when landmark detection fails.
    """
    h, w, _ = image.shape
    return {
        "Eyes::Left": image[int(h*0.2):int(h*0.35), int(w*0.2):int(w*0.4)],
        "Eyes::Right": image[int(h*0.2):int(h*0.35), int(w*0.6):int(w*0.8)],
        "Mouth::General": image[int(h*0.65):int(h*0.85), int(w*0.3):int(w*0.7)],
        "Nose::General": image[int(h*0.4):int(h*0.6), int(w*0.45):int(w*0.55)]
    }

def extract_rois(results_list):
    """
    Extract ROIs from all frames using landmarks or fallback.
    """
    all_rois = []

    for i, res in enumerate(results_list):
        image = res["image"]
        landmarks = res["landmarks"]

        #if landmarks:
        #    rois = extract_rois_from_landmarks(image, landmarks)
        #else:
        #    print(f"⚠️ Fallback ROI used for frame {i}")
        #    rois = fallback_rois(image)

        # 🚨 Temporarily force fallback ROIs only
        print(f"⚠️ Using fallback ROIs for frame {i} (bypassing landmark mapping)")
        rois = fallback_rois(image)

        all_rois.append(rois)

    return all_rois
