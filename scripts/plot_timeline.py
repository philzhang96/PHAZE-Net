# scripts/plot_timeline.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_timeline_from_csv(csv_path, video_id=None, output_path=None):
    df = pd.read_csv(csv_path)

    if video_id is None:
        video_id = os.path.splitext(os.path.basename(csv_path))[0].replace("_predictions", "")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"Emotion + VA Timeline — {video_id}", fontsize=14)

    axes[0].plot(df["frame"], df["transitioned_emotion"], drawstyle='default', marker='o', linestyle='-', color='tab:blue')
    axes[0].set_ylabel("Emotion")
    axes[0].tick_params(axis='y', labelrotation=45)
    axes[0].grid(True)

    axes[1].plot(df["frame"], df["smoothed_valence"], label="Smoothed Valence", color="tab:red")
    axes[1].plot(df["frame"], df["smoothed_arousal"], label="Smoothed Arousal", color="tab:green")
    axes[1].set_ylabel("Valence / Arousal")
    axes[1].set_xlabel("Frame")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), f"{video_id}_timeline.png")

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Saved timeline plot to {output_path}")
