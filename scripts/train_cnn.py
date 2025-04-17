import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets.affectnet_dataset import AffectNetDataset
from models.emotion_cnn import EmotionCNN

if __name__ == "__main__":
    # ---- CONFIG ----
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    ALPHA = 0.5  # VA loss weight
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- DEVICE ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- DATA ----
    dataset = AffectNetDataset(
        image_dir=r"E:\AffectNet\train_set\images",
        annotation_dir=r"E:\AffectNet\train_set\annotations"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ---- MODEL ----
    model = EmotionCNN(num_emotions=8).to(device)
    criterion_class = nn.CrossEntropyLoss()
    criterion_va = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- TRAINING LOOP ----
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_cls, total_va = 0.0, 0.0, 0.0

        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["emotion"].to(device)
            vals = batch["valence"].to(device)
            aros = batch["arousal"].to(device)

            optimizer.zero_grad()
            logits, va = model(imgs)

            loss_class = criterion_class(logits, labels)
            loss_va = criterion_va(va, torch.stack((vals, aros), dim=1))
            loss = loss_class + ALPHA * loss_va

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_class.item()
            total_va += loss_va.item()

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {total_loss:.4f} | Class: {total_cls:.4f} | VA: {total_va:.4f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"cnn_epoch{epoch+1}.pth"))

    print("✅ Training complete.")