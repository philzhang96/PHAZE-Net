import os
import cv2
from tqdm import tqdm
from config import FRAME_RATE
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import detect_faces_and_landmarks
from preprocessing.roi_extractor import extract_rois
from preprocessing.data_cleaner import clean_roi_batch

# ----- CONFIGURATION -----
AFEW_PATH = r"E:\PhD Datasets\AFEW\EmotiW_2018\Train_AFEW"  # Adjust as needed
OUTPUT_DIR = "data/cnn_training_data"
USE_CLEANING = False  # Set to True if you want to filter crops

def save_crops(emotion_label, video_path):
    try:
        frames = extract_frames(video_path, frame_rate=FRAME_RATE)
        if not frames:
            print(f"⚠️ Skipped {video_path} — no frames extracted.")
            return

        results = detect_faces_and_landmarks(frames, draw=False)
        rois_per_frame = extract_rois(results)

        for i, rois in enumerate(rois_per_frame):
            if USE_CLEANING:
                rois = clean_roi_batch(rois)
            for region, crop in rois.items():
                safe_region = region.replace("::", "_").replace(" ", "_")
                region_dir = os.path.join(OUTPUT_DIR, emotion_label, safe_region)
                os.makedirs(region_dir, exist_ok=True)
                save_path = os.path.join(region_dir, f"{os.path.basename(video_path)}_f{i:03d}.jpg")
                if crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0:
                    cv2.imwrite(save_path, crop)
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")

def process_afew_dataset():
    print(f"📁 Processing AFEW dataset in: {AFEW_PATH}")
    for emotion in os.listdir(AFEW_PATH):
        emotion_path = os.path.join(AFEW_PATH, emotion)
        if not os.path.isdir(emotion_path):
            continue
        print(f"\n🧠 Emotion: {emotion}")
        video_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(".avi")]
        for video_file in tqdm(video_files, desc=f"{emotion}", unit="video"):
            video_path = os.path.join(emotion_path, video_file)
            save_crops(emotion, video_path)

if __name__ == "__main__":
    process_afew_dataset()
    print("\n✅ Finished saving ROI crops for CNN training.")
