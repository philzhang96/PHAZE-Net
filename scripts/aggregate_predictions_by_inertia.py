import os
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
predictions_folder = r"data/pipeline_outputs"         # Folder with *_predictions.csv files
output_file = r"data/video_predictions.csv"           # Where final results will be saved
csv_suffix = "_predictions.csv"

# --- AGGREGATION ---
results = []

for fname in os.listdir(predictions_folder):
    if not fname.endswith(csv_suffix):
        continue

    video_id = fname.replace(csv_suffix, "")
    prediction_path = os.path.join(predictions_folder, fname)

    try:
        df = pd.read_csv(prediction_path)
    except Exception as e:
        print(f"❌ Error reading {fname}: {e}")
        continue

    if "transitioned_emotion" not in df.columns:
        print(f"⚠️ Skipping {video_id}: no 'transitioned_emotion' column found.")
        continue

    emotion_counts = Counter(df["transitioned_emotion"])
    most_common_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Unknown"

    results.append({
        "video_id": video_id,
        "predicted_emotion": most_common_emotion
    })

# --- SAVE ---
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
results_df.to_csv(output_file, index=False)

print(f"✅ Aggregated predictions saved to '{output_file}'")
