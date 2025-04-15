import cv2
import os

def extract_frames(video_path, frame_rate=1, output_dir=None):
    """
    Extract frames from a video at a specified frame rate (frames per second).

    Args:
        video_path (str): Path to the input video file.
        frame_rate (int): Number of frames to extract per second.
        output_dir (str, optional): Directory to save frames (as images). If None, frames are not saved.

    Returns:
        frames (list): List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)

    frames = []
    frame_idx = 0
    save_count = 0

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frames.append(frame)
            if output_dir:
                filename = os.path.join(output_dir, f"frame_{save_count:05d}.jpg")
                cv2.imwrite(filename, frame)
            save_count += 1

        frame_idx += 1

    cap.release()
    return frames
