import os
import numpy as np

from utils.video_processing import extract_frames
from utils.face_extraction import extract_faces_from_frames
from utils.feature_extraction import extract_features

data_path = "data"
save_path = "processed_data"

os.makedirs(save_path, exist_ok=True)

for label_folder in ["real", "fake"]:
    folder = os.path.join(data_path, label_folder)

    for video in os.listdir(folder):

        # ✅ Skip non-video files
        if not video.endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(folder, video)
        name = video.split(".")[0]

        frame_dir = f"temp_frames/{name}"
        face_dir = f"temp_faces/{name}"

        print(f"Processing: {video_path}")

        extract_frames(video_path, frame_dir)

        # ✅ Check if frames exist
        if not os.path.exists(frame_dir):
            print("Skipping, no frames extracted")
            continue

        extract_faces_from_frames(frame_dir, face_dir)

        features = extract_features(face_dir)

        if len(features) == 0:
            print("Skipping, no features")
            continue

        features = np.array(features)
        label = 0 if label_folder == "real" else 1

        np.save(f"{save_path}/{name}.npy", features)
        np.save(f"{save_path}/{name}_label.npy", label)

print("✅ Preprocessing Done")