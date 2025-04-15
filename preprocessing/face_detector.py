import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_faces_and_landmarks(frames, draw=True):
    """
    Detects face landmarks in a list of frames using MediaPipe Face Mesh.

    Args:
        frames (list): List of image frames (np.array format).
        draw (bool): Whether to draw landmarks on the images for inspection.

    Returns:
        results_list (list): List of dicts containing:
            - 'landmarks': list of (x, y) coords
            - 'image': optionally annotated image
    """
    results_list = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            output = {"landmarks": None, "image": frame.copy()}
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                coords = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in landmarks.landmark]
                output["landmarks"] = coords

                if draw:
                    for x, y in coords:
                        cv2.circle(output["image"], (x, y), 1, (0, 255, 0), -1)

            results_list.append(output)
    
    return results_list
