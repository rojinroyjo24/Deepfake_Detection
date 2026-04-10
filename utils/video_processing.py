import cv2
import os

# def extract_frames(video_path, output_folder, frame_skip=15, max_frames=20):
def extract_frames(video_path, output_folder, frame_skip=10, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video:", video_path)
        return

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        if count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"{video_path} → {saved} frames extracted")