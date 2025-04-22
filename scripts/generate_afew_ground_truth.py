import os
import csv

# 🔧 CHANGE THIS TO YOUR LOCAL PATH
afew_root = r"C:\PhD Datasets\AFEW\EmotiW_2018\Val_AFEW"
output_path = r"data/ground_truth.csv"

data = []
for emotion_dir in os.listdir(afew_root):
    emotion_path = os.path.join(afew_root, emotion_dir)
    if not os.path.isdir(emotion_path):
        continue

    for file in os.listdir(emotion_path):
        if file.endswith(".avi"):
            video_id = os.path.splitext(file)[0]
            data.append({"video_id": video_id, "ground_truth_emotion": emotion_dir})

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video_id", "ground_truth_emotion"])
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Wrote {len(data)} ground truth labels to {output_path}")
