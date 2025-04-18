import os
import torch
from torchvision import transforms
from PIL import Image
from models.emotion_cnn import EmotionCNN

# === CONFIG ===
folder = r"E:\PHAZE-Net\data\cnn_training_data\Angry\Cheeks_Left_cheekbone"  # Adjust as needed
checkpoint = r"E:\PHAZE-Net\checkpoints\cnn_epoch10.pth"
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = EmotionCNN(feature_dim=128)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval().to(device)

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# === LOAD IMAGES ===
images = []
for file in sorted(os.listdir(folder))[:15]:  # Limit to 15 for test
    if file.endswith(".jpg"):
        path = os.path.join(folder, file)
        img = Image.open(path).convert("RGB")
        images.append(transform(img))

input_tensor = torch.stack(images).to(device)  # Shape: [15, 3, 64, 64]

# === EXTRACT FEATURES ===
with torch.no_grad():
    features = model.extract_features(input_tensor)  # [15, 256]

print("✅ Feature tensor shape:", features.shape)
