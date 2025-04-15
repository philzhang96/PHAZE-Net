import os
import cv2
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import detect_faces_and_landmarks
from config import FRAME_RATE

def test_face_detection():
    video_path = "data/input_videos/sample_video.mp4"
    output_dir = "data/face_detection_preview"

    if not os.path.exists(video_path):
        print("❌ Sample video not found.")
        return

    frames = extract_frames(video_path, frame_rate=FRAME_RATE)
    results = detect_faces_and_landmarks(frames, draw=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save only successful landmarked frames (up to 10 for preview)
    saved = 0
    for i, res in enumerate(results):
        if res["landmarks"] is not None:
            out_path = os.path.join(output_dir, f"landmarked_frame_{i:05d}.jpg")
            cv2.imwrite(out_path, res["image"])
            saved += 1
        if saved >= 10:  # limit to 10 to keep preview light
            break

    print(f"✅ Landmarks detected in {sum(r['landmarks'] is not None for r in results)} of {len(frames)} frames.")
    print(f"🖼️ Saved {saved} annotated preview images (landmarked only) to '{output_dir}'.")

if __name__ == "__main__":
    test_face_detection()
