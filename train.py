import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from models.resnext_lstm import ResNeXtLSTM

# Initialize model
model = ResNeXtLSTM()

# Loss & optimizer (improved learning rate)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

data_path = "processed_data"

epochs = 10
best_loss = float("inf")

for epoch in range(epochs):
    total_loss = 0

    # ✅ Shuffle data
    files = [f for f in os.listdir(data_path) if f.endswith(".npy") and "label" not in f]
    random.shuffle(files)

    for file in files:
        feature_path = os.path.join(data_path, file)
        label_path = feature_path.replace(".npy", "_label.npy")

        # Load data
        features = np.load(feature_path)
        label = int(np.load(label_path))

        # Convert to tensor
        features_tensor = torch.tensor(features).unsqueeze(0).float()
        label_tensor = torch.tensor([[label]]).float()

        # Forward
        output = model(features_tensor)

        # Loss
        loss = criterion(output, label_tensor)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ✅ Save best model
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved")

print("🎯 Training Completed")