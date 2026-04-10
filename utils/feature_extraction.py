import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Updated model loading
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_folder, max_images=20):
    features_list = []

    for i, file in enumerate(os.listdir(image_folder)):
        if i >= max_images:
            break

        if file.endswith(".jpg"):
            path = os.path.join(image_folder, file)
            img = Image.open(path).convert("RGB")

            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                features = model(tensor)

            features = features.flatten().numpy()
            features_list.append(features)

    return features_list