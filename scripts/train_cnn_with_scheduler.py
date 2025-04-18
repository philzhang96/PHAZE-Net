
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets.affectnet_dataset import AffectNetDataset
from models.emotion_cnn import EmotionCNN

# --- CONFIGURATION ---
ANNOTATION_DIR = r"E:\AffectNet\train_set\annotations"
IMAGE_DIR = r"E:\AffectNet\train_set\images"
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_EVERY = 5
EARLY_STOP_PATIENCE = 5

# --- DATA LOADING ---
dataset = AffectNetDataset(annotation_dir=ANNOTATION_DIR, image_dir=IMAGE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_emotions=8, feature_dim=256).to(device)

# --- OPTIMIZER & SCHEDULER ---
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
criterion_cls = nn.CrossEntropyLoss()
criterion_va = nn.MSELoss()

# --- LOAD CHECKPOINT IF EXISTS ---
start_epoch = 1
best_loss = float('inf')
no_improve_epochs = 0
resume_path = os.path.join(CHECKPOINT_DIR, "emotion_cnn_epoch10.pth")
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', best_loss)
    print(f"🔁 Resumed training from epoch {start_epoch}")

# --- CHECKPOINT FUNCTION ---
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# --- TRAINING LOOP ---
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    model.train()
    total_loss, total_cls_loss, total_va_loss = 0, 0, 0

    for images, labels, vals, aros in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        vals = vals.to(device)
        aros = aros.to(device)

        optimizer.zero_grad()
        logits, va = model(images)

        loss_cls = criterion_cls(logits, labels)
        loss_va = criterion_va(va, torch.stack((vals, aros), dim=1))
        loss = loss_cls + 0.1 * loss_va

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_va_loss += loss_va.item()

    avg_loss = total_loss / len(dataloader)
    avg_cls = total_cls_loss / len(dataloader)
    avg_va = total_va_loss / len(dataloader)
    print(f"[Epoch {epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Class: {avg_cls:.4f} | VA: {avg_va:.4f}")

    # --- LR Scheduler Step ---
    scheduler.step(avg_loss)

    # --- Early Stopping Check ---
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_epochs = 0
        save_checkpoint(epoch, model, optimizer, avg_loss, os.path.join(CHECKPOINT_DIR, "emotion_cnn_best.pth"))
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= EARLY_STOP_PATIENCE:
        print("⛔ Early stopping triggered.")
        break

    # --- Periodic Checkpoint ---
    if epoch % CHECKPOINT_EVERY == 0:
        save_checkpoint(epoch, model, optimizer, avg_loss, os.path.join(CHECKPOINT_DIR, f"emotion_cnn_epoch{epoch}.pth"))

print("✅ Training complete.")
