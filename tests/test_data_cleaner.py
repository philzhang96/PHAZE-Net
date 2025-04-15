import os
import cv2
import numpy as np
from preprocessing.data_cleaner import clean_roi_batch

def generate_test_crops():
    good = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    small = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    dark = np.zeros((50, 50, 3), dtype=np.uint8)
    bright = np.full((50, 50, 3), 255, dtype=np.uint8)
    flat = np.full((50, 50, 3), 127, dtype=np.uint8)
    noisy = np.random.normal(127, 1, (50, 50, 3)).astype(np.uint8)

    return {
        "Eyes::Left": [good, small, dark, bright, flat, noisy],
        "Mouth::Upper lip (upper)": [flat, good, bright, None]
    }

def test_cleaning_and_save():
    output_dir = "data/cleaned_roi_preview"
    os.makedirs(output_dir, exist_ok=True)

    test_rois = generate_test_crops()
    print("🧪 Testing ROI Cleaner...")

    cleaned = clean_roi_batch(test_rois, verbose=True)

    count = 0
    for region, crops in cleaned.items():
        safe_region = region.replace("::", "_").replace(" ", "_")
        for i, crop in enumerate(crops):
            out_path = os.path.join(output_dir, f"{safe_region}_{i:02d}.jpg")
            cv2.imwrite(out_path, crop)
            count += 1

    print(f"\n🖼️ Saved {count} cleaned ROI crops to '{output_dir}'")
    print("🧼 ROI cleaning and visualisation test completed.")

if __name__ == "__main__":
    test_cleaning_and_save()
