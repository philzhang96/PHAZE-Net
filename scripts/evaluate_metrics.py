import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
import os

# --- CONFIGURATION ---
prediction_file = r"data/video_predictions.csv"
ground_truth_file = r"data/ground_truth.csv"
va_gt_file = r"data/va_ground_truth.csv"
perframe_pred_dir = r"data/pipeline_outputs"
output_file = r"data/evaluation_metrics.txt"

# --- METRIC: CCC ---
def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc

# --- PART 1: Classification Metrics ---
pred_df = pd.read_csv(prediction_file)
gt_df = pd.read_csv(ground_truth_file)
df = pd.merge(gt_df, pred_df, on="video_id", how="inner")

y_true = df["ground_truth_emotion"]
y_pred = df["predicted_emotion"]

report = classification_report(y_true, y_pred, digits=3)
conf_matrix = confusion_matrix(y_true, y_pred)
top1_acc = accuracy_score(y_true, y_pred)

# --- PART 2: VA CCC and MSE (only if available) ---
valence_ccc = arousal_ccc = valence_mse = arousal_mse = None  # Init defaults

if os.path.exists(va_gt_file):
    va_gt = pd.read_csv(va_gt_file)

    valence_true_all = []
    valence_pred_all = []
    arousal_true_all = []
    arousal_pred_all = []

    for video_id in va_gt["video_id"].unique():
        va_gt_vid = va_gt[va_gt["video_id"] == video_id]
        pred_path = os.path.join(perframe_pred_dir, f"{video_id}_predictions.csv")

        if not os.path.exists(pred_path):
            print(f"⚠️ Missing prediction file: {pred_path}")
            continue

        pred_df = pd.read_csv(pred_path)
        merged = pd.merge(va_gt_vid, pred_df, on="frame", how="inner")

        valence_true_all.extend(merged["valence"].tolist())
        valence_pred_all.extend(merged["smoothed_valence"].tolist())
        arousal_true_all.extend(merged["arousal"].tolist())
        arousal_pred_all.extend(merged["smoothed_arousal"].tolist())

    if valence_true_all:
        valence_ccc = concordance_correlation_coefficient(valence_true_all, valence_pred_all)
        arousal_ccc = concordance_correlation_coefficient(arousal_true_all, arousal_pred_all)
        valence_mse = mean_squared_error(valence_true_all, valence_pred_all)
        arousal_mse = mean_squared_error(arousal_true_all, arousal_pred_all)

# --- SAVE RESULTS ---
with open(output_file, "w", encoding="utf-8") as f:
    f.write("📊 Classification Report\n")
    f.write(report)
    f.write("\n\n🧩 Confusion Matrix\n")
    f.write(str(conf_matrix))
    f.write("\n\n🎯 Top-1 Accuracy: {:.3f}\n".format(top1_acc))

    if valence_ccc is not None:
        f.write("\n\n💡 Valence CCC: {:.3f}\n".format(valence_ccc))
        f.write("💡 Arousal CCC: {:.3f}\n".format(arousal_ccc))
        f.write("\n📉 Valence MSE: {:.4f}\n".format(valence_mse))
        f.write("📉 Arousal MSE: {:.4f}\n".format(arousal_mse))
    else:
        f.write("\n\n⚠️ Skipped VA metrics — no AffectNet predictions or VA ground truth found.\n")

print(f"✅ Evaluation complete. Results saved to '{output_file}'")
