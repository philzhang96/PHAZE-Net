import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks_from_video(video_path, save_path, draw=False, resize=None):
    """
    Extracts facial landmarks from a video using MediaPipe Face Mesh and saves as .npy.

    Args:
        video_path (str): Path to input video.
        save_path (str): Path to save .npy file containing landmarks (T, 468, 2).
        draw (bool): Whether to annotate frames for visualisation (not saved).
        resize (tuple): Optional (width, height) to resize frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()

    landmark_sequence = []
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        for frame in tqdm(frames, desc=f"Processing {os.path.basename(video_path)}"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                coords = [(p.x * frame.shape[1], p.y * frame.shape[0]) for p in landmarks.landmark]
                landmark_sequence.append(coords)
            else:
                # If no face is detected, append NaNs
                landmark_sequence.append([(np.nan, np.nan)] * 468)

    # Filter out frames where landmarks are None or incomplete
    valid_landmarks = [l for l in landmark_sequence if isinstance(l, (list, tuple)) and len(l) == 468]
    landmark_array = np.array(valid_landmarks)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, landmark_array)
    print(f"Saved landmark sequence to: {save_path}")
