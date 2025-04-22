import os
import shutil
from tqdm import tqdm

def flatten_afew_structure(root_dir, output_dir):
    """
    Flattens the directory structure:
    From: root_dir/emotion/region/image.jpg
    To:   output_dir/emotion/image.jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    emotions = os.listdir(root_dir)

    for emotion in tqdm(emotions, desc="Processing emotions"):
        emotion_path = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        # Destination directory
        flat_emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(flat_emotion_dir, exist_ok=True)

        counter = 0
        for region in os.listdir(emotion_path):
            region_path = os.path.join(emotion_path, region)
            if not os.path.isdir(region_path):
                continue

            for filename in os.listdir(region_path):
                src = os.path.join(region_path, filename)
                ext = os.path.splitext(filename)[-1].lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Rename to avoid overwriting: emotion_region_counter.jpg
                dst_filename = f"{emotion}_{region}_{counter:05d}{ext}"
                dst = os.path.join(flat_emotion_dir, dst_filename)
                shutil.copy(src, dst)
                counter += 1

    print("✅ Flattening complete.")

# Example usage
if __name__ == "__main__":
    root = r"C:\PHAZE-Net\data\cnn_training_data"
    output = r"C:\PhD Datasets\AFEW\cnn_flattened"
    flatten_afew_structure(root, output)
