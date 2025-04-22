import os
import subprocess

# --- CONFIGURATION ---
video_dir = r"E:\PhD Datasets\AFEW\EmotiW_2018\Val_AFEW"  # Folder with all .avi test videos
video_ext = ".avi"                        # Video extension to look for
main_script = "main.py"                  # Entry point for inference
video_arg = "--video"                    # Argument to pass into main.py

# --- DISCOVER VIDEOS ---
video_list = []
for root, dirs, files in os.walk(video_dir):
    for f in files:
        if f.endswith(video_ext):
            video_list.append(os.path.join(root, f))

print(f"🔍 Found {len(video_list)} videos in '{video_dir}' and subfolders")


# --- RUN INFERENCE PER VIDEO ---
for idx, video_file in enumerate(video_list):
    video_path = os.path.join(video_dir, video_file)
    print(f"\n[{idx+1}/{len(video_list)}] 🚀 Running inference on: {video_file}")

    try:
        subprocess.run(
            ["python", main_script, video_arg, video_path],
            check=True
        )
        print(f"✅ Done: {video_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed on {video_file}: {e}")
