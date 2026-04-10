# from utils.video_processing import extract_frames
# from utils.face_extraction import extract_faces_from_frames
# from utils.feature_extraction import extract_features

# import numpy as np
# import torch
# from models.resnext_lstm import ResNeXtLSTM
# import os

# # ==============================
# # 🔥 INPUT: Change video here
# # ==============================
# video_path = "test_videos/f2.mp4"

# # Automatically extract video name
# video_name = os.path.splitext(os.path.basename(video_path))[0]

# # Temp folders
# frame_dir = f"temp_frames/{video_name}"
# face_dir = f"temp_faces/{video_name}"

# # ==============================
# # 🧹 Clean old data (optional)
# # ==============================
# import shutil

# if os.path.exists(frame_dir):
#     shutil.rmtree(frame_dir)

# if os.path.exists(face_dir):
#     shutil.rmtree(face_dir)

# # ==============================
# # 🎥 STEP 1: Extract frames
# # ==============================
# print("Extracting frames...")
# extract_frames(video_path, frame_dir)

# # ==============================
# # 🙂 STEP 2: Extract faces
# # ==============================
# print("Extracting faces...")
# extract_faces_from_frames(frame_dir, face_dir)

# # ==============================
# # 🧠 STEP 3: Extract features
# # ==============================
# print("Extracting features...")
# features = extract_features(face_dir)

# if len(features) == 0:
#     print("❌ No faces detected. Try another video.")
#     exit()

# features_np = np.array(features)
# features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # ==============================
# # 🤖 STEP 4: Load model
# # ==============================
# model = ResNeXtLSTM()

# # Use best trained model
# model.load_state_dict(torch.load("best_model.pth"))
# model.eval()

# # ==============================
# # 🔍 STEP 5: Prediction
# # ==============================
# output = model(features_tensor)
# prob = torch.sigmoid(output)

# print("\n🎯 RESULT")
# print("Probability:", round(prob.item(), 4))

# if prob.item() > 0.5:
#     print("Prediction: FAKE 🚨")
# else:
#     print("Prediction: REAL ✅")

def predict_video(video_path):
    from utils.video_processing import extract_frames
    from utils.face_extraction import extract_faces_from_frames
    from utils.feature_extraction import extract_features

    import numpy as np
    import torch
    from models.resnext_lstm import ResNeXtLSTM
    import os, shutil

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_dir = f"temp_frames/{video_name}"
    face_dir = f"temp_faces/{video_name}"

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    extract_frames(video_path, frame_dir)
    extract_faces_from_frames(frame_dir, face_dir)

    features = extract_features(face_dir)

    if len(features) == 0:
        return "No Face Detected", 0

    features = np.array(features)
    features_tensor = torch.tensor(features).unsqueeze(0).float()

    model = ResNeXtLSTM()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    output = model(features_tensor)
    prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "FAKE", prob
    else:
        return "REAL", prob