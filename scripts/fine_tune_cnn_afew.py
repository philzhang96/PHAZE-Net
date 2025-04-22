import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from datasets.folder_flattened_dataset import FlattenedEmotionDataset
from models.emotion_cnn import EmotionCNN

# --- CONFIGURATION ---
DATA_DIR = r"C:\PhD Datasets\AFEW\cnn_flattened"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 1
BATCH_SIZE = 512
NUM_EPOCHS = 10000
LEARNING_RATE = 5e-5
NUM_WORKERS = 17
EARLY_STOP_PATIENCE = 30
PRETRAINED_CKPT = os.path.join(CHECKPOINT_DIR, "emotion_cnn_afew_epoch100")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    print("🚀 Starting fine-tuning on AFEW")

    # --- Dataset ---
    dataset = FlattenedEmotionDataset(DATA_DIR)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_emotions=7, feature_dim=256).to(device)

    # Load pretrained AffectNet weights and modify classifier
    if os.path.exists(PRETRAINED_CKPT):
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
        print(f"🔁 Loaded pretrained weights from {PRETRAINED_CKPT}")
        state_dict = ckpt['model_state_dict']
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        model.load_state_dict(state_dict, strict=False)
        state_dict.pop("classifier_head.2.weight", None)
        state_dict.pop("classifier_head.2.bias", None)
        model.load_state_dict(state_dict, strict=False)

    for param in model.backbone.parameters():
        param.requires_grad = False  # Freeze backbone for fine-tuning

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits, _ = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Fine-tune Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "emotion_cnn_afew_best.pth"))
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print("⛔ Early stopping triggered.")
            break

        # Save every epoch
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"emotion_cnn_afew_epoch{epoch}.pth"))

    print("✅ Fine-tuning complete.")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
