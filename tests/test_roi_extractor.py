import os
import cv2
from config import FRAME_RATE
from preprocessing.frame_extractor import extract_frames
from preprocessing.face_detector import detect_faces_and_landmarks
from preprocessing.roi_extractor import extract_rois

def test_roi_extraction():
    video_path = "data/input_videos/sample_video.mp4"
    output_dir_base = "data/roi_preview"
    fallback_dir = os.path.join(output_dir_base, "fallback")
    landmark_dir = os.path.join(output_dir_base, "landmark")

    for d in [fallback_dir, landmark_dir]:
        os.makedirs(d, exist_ok=True)

    frames = extract_frames(video_path, frame_rate=FRAME_RATE)
    results = detect_faces_and_landmarks(frames, draw=False)
    rois_list = extract_rois(results)

    fallback_saved = 0
    landmark_saved = 0

    for frame_idx, (res, rois) in enumerate(zip(results, rois_list)):
        is_fallback = res["landmarks"] is None
        target_dir = fallback_dir if is_fallback else landmark_dir

        for roi_name, roi_img in rois.items():
            safe_name = roi_name.replace("::", "_").replace(" ", "_")
            out_path = os.path.join(target_dir, f"frame{frame_idx:03d}_{safe_name}.jpg")

            if roi_img.shape[0] > 0 and roi_img.shape[1] > 0:
                cv2.imwrite(out_path, roi_img)

        if is_fallback:
            fallback_saved += 1
        else:
            landmark_saved += 1

        # Only preview a few from each type
        if fallback_saved >= 3 and landmark_saved >= 3:
            break

    print(f"✅ Saved ROI previews: {landmark_saved} with landmarks, {fallback_saved} fallback crops.")
    print(f"📂 Check: {landmark_dir} and {fallback_dir}")

if __name__ == "__main__":
    test_roi_extraction()
