import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
sys.path.insert(0, os.path.dirname(__file__))
from datasets import Data3Dataset
from module import UnetWithLTAE
import random, numpy as np
from tqdm import tqdm

# ===== Config =====
DATA_DIR = os.path.join(os.path.dirname(__file__), "train_data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 2
LR = 1e-4
NUM_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

def compute_dice(logits, targets, threshold=0.5, eps=1e-5):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    return ((2. * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean().item()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    set_seed(42)

    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    print(f"Total samples: {len(dataset)}")

    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = BceDiceLoss()

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += compute_dice(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ✓ Saved best model (epoch {epoch+1})")

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

    print("Training complete!")
