import os
import cv2
from config import FRAME_RATE
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import detect_faces_and_landmarks
from preprocessing.roi_extractor import extract_rois
from preprocessing.data_cleaner import clean_roi_batch
from preprocessing.temporal_chunker import multi_resolution_chunking

# ----- CONFIGURATION -----
video_path = r"E:\PhD Datasets\AFEW\EmotiW_2018\Test\Test_vid_Distribute\Test_vid_Distribute\000029320.avi"
output_dir = "data/pipeline_outputs"
chunk_preview_dir = os.path.join(output_dir, "chunk_preview")
os.makedirs(chunk_preview_dir, exist_ok=True)

chunk_config = {
    "short":  {"size": 8, "stride": 2},  # 👈 Try smaller chunks
    "medium": {"size": 12, "stride": 3},
    "long":   {"size": 14, "stride": 4}
}

# ----- STEP 1: Frame Extraction -----
print("🎞️ Extracting frames...")
frames = extract_frames(video_path, frame_rate=FRAME_RATE)
print(f"✅ Extracted {len(frames)} frames.")

# ----- STEP 2: Face & Landmark Detection -----
print("🧠 Detecting faces and landmarks...")
results = detect_faces_and_landmarks(frames, draw=False)
num_landmarked = sum(r["landmarks"] is not None for r in results)
print(f"✅ Landmarks detected in {num_landmarked} of {len(results)} frames.")

# ----- STEP 3: ROI Extraction -----
print("📦 Extracting ROIs...")
rois_per_frame = extract_rois(results)

# ----- STEP 4: ROI Cleaning -----
print("🧼 Cleaning ROIs...")
cleaned_rois_all = []
for rois in rois_per_frame:
    cleaned = clean_roi_batch(rois)
    cleaned_rois_all.append(cleaned)

# ----- STEP 5: Sequence Assembly -----
print("🔁 Combining ROIs by region...")
roi_sequences = {}
for frame_dict in cleaned_rois_all:
    for region, crop in frame_dict.items():
        if region not in roi_sequences:
            roi_sequences[region] = []
        roi_sequences[region].append(crop)

# NEW: Print how many valid frames each region has
print("\n📊 Valid frames per region:")
for region, crops in roi_sequences.items():
    print(f" - {region}: {len(crops)} frames")

# ----- STEP 6: Temporal Chunking -----
print("⏱️ Chunking ROI sequences...")
chunked_output = multi_resolution_chunking(roi_sequences, chunk_config)

# ----- STEP 7: Save Previews of Chunked Output -----
print("💾 Saving preview crops from 'short' window chunks...")
saved = 0
for region, scales in chunked_output.items():
    short_chunks = scales.get("short", [])[:2]  # preview first 2 chunks only
    for i, chunk in enumerate(short_chunks):
        for j, crop in enumerate(chunk[:3]):  # only save first 3 crops per chunk
            safe_region = region.replace("::", "_").replace(" ", "_")
            path = os.path.join(chunk_preview_dir, f"{safe_region}_chunk{i}_frame{j}.jpg")
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                cv2.imwrite(path, crop)
                saved += 1

print(f"✅ Pipeline complete. Saved {saved} chunk preview crops.")
