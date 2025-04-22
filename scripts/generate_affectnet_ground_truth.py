import os
import numpy as np
import csv
from glob import glob

# --- CONFIGURATION ---
annotations_dir = r"C:\AffectNet\val_set\annotations"
va_output_path = r"data/va_ground_truth.csv"
discrete_output_path = r"data/affectnet_discrete_labels.csv"

# --- Emotion Mapping ---
emotion_map = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"
}

# --- Find all *_val.npy files ---
val_files = sorted(glob(os.path.join(annotations_dir, "*_val.npy")))
total = len(val_files)

os.makedirs(os.path.dirname(va_output_path), exist_ok=True)
with open(va_output_path, "w", newline="") as f_va, open(discrete_output_path, "w", newline="") as f_exp:
    va_writer = csv.DictWriter(f_va, fieldnames=["video_id", "frame", "valence", "arousal"])
    exp_writer = csv.DictWriter(f_exp, fieldnames=["image_id", "ground_truth_emotion"])
    va_writer.writeheader()
    exp_writer.writeheader()

    for val_file in val_files:
        idx = os.path.basename(val_file).split("_")[0]
        try:
            val = float(np.load(os.path.join(annotations_dir, f"{idx}_val.npy")))
            aro = float(np.load(os.path.join(annotations_dir, f"{idx}_aro.npy")))
            exp = int(np.load(os.path.join(annotations_dir, f"{idx}_exp.npy")))
            emotion = emotion_map.get(exp, "Unknown")

            va_writer.writerow({
                "video_id": "AffectNet",
                "frame": int(idx),
                "valence": val,
                "arousal": aro
            })

            exp_writer.writerow({
                "image_id": int(idx),
                "ground_truth_emotion": emotion
            })

        except Exception as e:
            print(f"⚠️ Skipping index {idx} due to error: {e}")

print(f"✅ Wrote {total} entries to {va_output_path}")
print(f"✅ Wrote {total} entries to {discrete_output_path}")
