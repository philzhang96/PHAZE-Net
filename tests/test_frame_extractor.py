import os
from preprocessing.frame_extractor import extract_frames
from config import FRAME_RATE

def test_extract_and_save_frames():
    video_path = "data/input_videos/sample_video.mp4"
    output_dir = "data/frames"

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    frames = extract_frames(video_path, frame_rate=FRAME_RATE, output_dir=output_dir)

    assert len(frames) > 0, "No frames were extracted."
    assert os.path.exists(output_dir), "Output directory was not created."
    assert os.path.isfile(os.path.join(output_dir, "frame_00000.jpg")), "No frames saved."

    print(f"✅ Test passed. {len(frames)} frames saved to '{output_dir}'.")

if __name__ == "__main__":
    test_extract_and_save_frames()
