import os
import cv2
import torch
from torchvision import transforms
from config import FRAME_RATE
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import detect_faces_and_landmarks
from preprocessing.roi_extractor import extract_rois
from preprocessing.temporal_chunker import multi_resolution_chunking
from models.emotion_cnn import EmotionCNN
from models.hierarchical_agfw import HierarchicalAGFW
from models.emotion_predictor import EmotionPredictor
from InERTIA.inertia_controller import InERTIAController
from InERTIA.region_aggregator import RegionAggregator
from InERTIA.smoother import Smoother

# ----- CONFIGURATION -----
video_path = r"E:\PhD Datasets\AFEW\EmotiW_2018\Test\Test_vid_Distribute\Test_vid_Distribute\000029320.avi"
output_dir = "data/pipeline_outputs"
os.makedirs(output_dir, exist_ok=True)

chunk_config = {
    "short": {"size": 3, "stride": 1},
    "medium": {"size": 4, "stride": 1},
    "long": {"size": 5, "stride": 1}
}

# ----- STEP 1: Frame Extraction -----
print("🎞️ Extracting frames...")
frames = extract_frames(video_path, frame_rate=FRAME_RATE)
print(f"✅ Extracted {len(frames)} frames.")

# ----- STEP 2: Face & Landmark Detection -----
print("🧠 Detecting faces and landmarks...")
results = detect_faces_and_landmarks(frames, draw=False)
print(f"✅ Landmarks detected in {sum(r['landmarks'] is not None for r in results)} frames.")

# ----- STEP 3–4: ROI Extraction (Cleaning skipped for now) -----
print("📦 Extracting ROIs...")
cleaned_rois_all = extract_rois(results)

# ----- STEP 5: Assemble ROI Sequences -----
roi_sequences = {}
for frame_dict in cleaned_rois_all:
    for region, crop in frame_dict.items():
        roi_sequences.setdefault(region, []).append(crop)
print(f"📊 Found {len(roi_sequences)} valid ROI regions.")

# ----- STEP 6: Temporal Chunking -----
print("⏱️ Chunking sequences...")
chunked_output = multi_resolution_chunking(roi_sequences, chunk_config)

# ----- STEP 7: Load Models -----
print("🧠 Loading trained models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_emotions=8, feature_dim=128).to(device)
model.load_state_dict(torch.load("checkpoints/cnn_epoch10.pth", map_location=device))
model.eval()

agfw = HierarchicalAGFW(input_dim=128).to(device)
predictor = EmotionPredictor(input_dim=128, num_emotions=8).to(device)
inertia = InERTIAController(momentum_alpha=0.8, transition_lambda=2.0)
aggregator = RegionAggregator()
smoother = Smoother(alpha=0.6)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ----- STEP 8: Multi-Chunk Rollout + Region Aggregation + Smoothing -----
print("🔄 Running multi-chunk emotion prediction with InERTIA...")
with torch.no_grad():
    region_names = list(chunked_output.keys())
    max_chunks = min(
        len(chunked_output[r]["short"]) for r in region_names
        if all(k in chunked_output[r] for k in ["short", "medium", "long"])
    )

    for i in range(max_chunks):
        region_outputs = {}
        confidences = {}

        print(f"\n🕒 Frame Segment {i+1} | Processing {len(region_names)} regions")

        for region in region_names:
            scales = chunked_output[region]
            if not all(k in scales and len(scales[k]) > i for k in ["short", "medium", "long"]):
                continue

            def chunk_to_tensor(chunk):
                return torch.stack([transform(img) for img in chunk]).unsqueeze(0).to(device)

            short = chunk_to_tensor(scales["short"][i])
            medium = chunk_to_tensor(scales["medium"][i])
            long = chunk_to_tensor(scales["long"][i])

            short_feat = model.extract_features(short.squeeze(0))
            medium_feat = model.extract_features(medium.squeeze(0))
            long_feat = model.extract_features(long.squeeze(0))

            fused, weights = agfw(short_feat.unsqueeze(0), medium_feat.unsqueeze(0), long_feat.unsqueeze(0))
            logits, va = predictor(fused)

            region_outputs[region] = fused.squeeze(0)
            confidences[region] = torch.stack([w.mean() for w in weights.values()]).mean().item()



        # Skip if nothing to aggregate
        if not region_outputs:
            continue

        aggregated = aggregator.aggregate(region_outputs, confidences, inertia.current_emotion)
        logits, va = predictor(aggregated.unsqueeze(0))

        predicted_class = torch.argmax(logits, dim=1).item()
        valence, arousal = va.squeeze().tolist()
        smoothed_va = smoother.update(valence, arousal)

        updated_emotion, changed = inertia.step(smoothed_va[0], smoothed_va[1])
        status = "🔁 Transitioned" if changed else "✅ Stayed"

        print(f"🎯 Aggregated Prediction → Class={predicted_class}, VA=({smoothed_va[0]:.2f}, {smoothed_va[1]:.2f}) → {updated_emotion} {status}")
