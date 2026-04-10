# # # # from utils.video_processing import extract_frames

# # # # extract_frames(
# # # #     "data/real/video1.mp4",
# # # #     "extracted_frames/sample_video",
# # # #     frame_rate=10
# # # # )

# # # # from utils.face_extraction import extract_faces_from_frames

# # # # extract_faces_from_frames(
# # # #     "extracted_frames/sample_video",
# # # #     "extracted_frames/faces"
# # # # )

# # # # from utils.feature_extraction import extract_features

# # # # features = extract_features("extracted_frames/faces")

# # # # print("Number of frames:", len(features))
# # # # print("Feature vector size:", len(features[0]))


# # # # import torch
# # # # from models.resnext_lstm import ResNeXtLSTM

# # # # # Convert features to tensor
# # # # features_tensor = torch.tensor(features).unsqueeze(0).float()

# # # # model = ResNeXtLSTM()

# # # # output = model(features_tensor)

# # # # print("Output:", output)


# # # # import torch
# # # # from models.resnext_lstm import ResNeXtLSTM

# # # # features_tensor = torch.tensor(features).unsqueeze(0).float()

# # # # model = ResNeXtLSTM()

# # # # output = model(features_tensor)

# # # # print("Output:", output)

# # # # from utils.feature_extraction import extract_features
# # # # import numpy as np
# # # # import torch
# # # # from models.resnext_lstm import ResNeXtLSTM

# # # # # Step 1: Extract features
# # # # features = extract_features("extracted_frames/faces")

# # # # # Step 2: Convert to tensor
# # # # features_np = np.array(features)
# # # # features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # # # # Step 3: Load model
# # # # model = ResNeXtLSTM()

# # # # # Step 4: Get output
# # # # output = model(features_tensor)

# # # # print("Output:", output)

# # # # from utils.feature_extraction import extract_features
# # # # import numpy as np
# # # # import torch
# # # # from models.resnext_lstm import ResNeXtLSTM

# # # # # Step 1: Extract features
# # # # features = extract_features("extracted_frames/faces")

# # # # # Step 2: Convert to tensor
# # # # features_np = np.array(features)
# # # # features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # # # # Step 3: Load model
# # # # model = ResNeXtLSTM()

# # # # # Step 4: Prediction
# # # # output = model(features_tensor)

# # # # print("Output:", output)

# # # # Step 0: Imports
# # # from utils.feature_extraction import extract_features
# # # import numpy as np
# # # import torch
# # # from models.resnext_lstm import ResNeXtLSTM

# # # # Step 1: Extract features from face images
# # # features = extract_features("extracted_frames/faces")

# # # # Safety check
# # # if len(features) == 0:
# # #     print("No features found. Check face extraction step.")
# # #     exit()

# # # print("Number of frames:", len(features))

# # # # Step 2: Convert features to tensor
# # # features_np = np.array(features)
# # # print("Feature shape:", features_np.shape)

# # # features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # # # Step 3: Load LSTM model
# # # model = ResNeXtLSTM()

# # # # Step 4: Get model output
# # # output = model(features_tensor)
# # # print("Raw Output:", output)

# # # # Step 5: Apply sigmoid to get probability
# # # prob = torch.sigmoid(output)
# # # print("Probability:", prob.item())

# # # # Step 6: Final prediction
# # # if prob.item() > 0.5:
# # #     print("Prediction: FAKE")
# # # else:
# # #     print("Prediction: REAL")

# # from utils.feature_extraction import extract_features
# # import numpy as np
# # import torch
# # from models.resnext_lstm import ResNeXtLSTM

# # # Step 1: Extract features from faces
# # features = extract_features("temp_faces/sample_video")

# # # Safety check
# # if len(features) == 0:
# #     print("No features found. Check preprocessing.")
# #     exit()

# # # Convert to tensor
# # features_np = np.array(features)
# # features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # # Load trained model
# # model = ResNeXtLSTM()
# # model.load_state_dict(torch.load("model.pth"))
# # model.eval()

# # # Prediction
# # output = model(features_tensor)

# # # Convert to probability
# # prob = torch.sigmoid(output)

# # print("Probability:", prob.item())

# # # Final classification
# # if prob.item() > 0.5:
# #     print("Prediction: FAKE")
# # else:
# #     print("Prediction: REAL")

# from utils.feature_extraction import extract_features
# import numpy as np
# import torch
# from models.resnext_lstm import ResNeXtLSTM
# import os

# # Get available face folders
# face_folders = os.listdir("temp_faces")

# if len(face_folders) == 0:
#     print("No face data found. Run preprocess.py first.")
#     exit()

# # Use first folder
# face_path = f"temp_faces/{face_folders[0]}"
# print("Using folder:", face_path)

# # Extract features
# features = extract_features(face_path)

# if len(features) == 0:
#     print("No features found.")
#     exit()

# # Convert to tensor
# features_np = np.array(features)
# features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# # Load model
# model = ResNeXtLSTM()
# model.load_state_dict(torch.load("model.pth"))
# model.eval()

# # Prediction
# output = model(features_tensor)
# prob = torch.sigmoid(output)

# print("Probability:", prob.item())

# if prob.item() > 0.5:
#     print("Prediction: FAKE")
# else:
#     print("Prediction: REAL")

from utils.feature_extraction import extract_features
import numpy as np
import torch
from models.resnext_lstm import ResNeXtLSTM
import os

# Get available face folders
face_folders = os.listdir("temp_faces")

if len(face_folders) == 0:
    print("No face data found. Run preprocess.py first.")
    exit()

# Use first folder (or change manually)
face_path = f"temp_faces/{face_folders[11]}"
print("Using folder:", face_path)

# Extract features
features = extract_features(face_path)

if len(features) == 0:
    print("No features found.")
    exit()

# Convert to tensor
features_np = np.array(features)
features_tensor = torch.tensor(features_np).unsqueeze(0).float()

# Load BEST trained model
model = ResNeXtLSTM()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Prediction
output = model(features_tensor)
prob = torch.sigmoid(output)

print("Probability:", prob.item())

# Final decision
if prob.item() > 0.5:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")