import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class AffectNetDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, include_va=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.include_va = include_va

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # sort by index

        # Image transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_id = os.path.splitext(filename)[0]

        img_path = os.path.join(self.image_dir, filename)
        exp_path = os.path.join(self.annotation_dir, f"{image_id}_exp.npy")
        val_path = os.path.join(self.annotation_dir, f"{image_id}_val.npy")
        aro_path = os.path.join(self.annotation_dir, f"{image_id}_aro.npy")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        image = self.transform(image)

        try:
            emotion = int(np.load(exp_path, allow_pickle=True).item())
            valence = np.float32(np.load(val_path, allow_pickle=True).item())
            arousal = np.float32(np.load(aro_path, allow_pickle=True).item())
        except Exception as e:
            raise RuntimeError(f"Failed to load annotation for {image_id}: {e}")

        if self.include_va:
            return {
                "image": image,
                "emotion": emotion,
                "valence": valence,
                "arousal": arousal
            }
        else:
            return {
                "image": image,
                "emotion": emotion
            }
