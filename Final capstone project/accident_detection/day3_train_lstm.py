"""
Day 3 - Part 2: LSTM Temporal Anomaly Classifier
Takes sequences of 16 frames → predicts accident or normal
Uses torchvision for feature extraction (ResNet18 backbone)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────
MANIFEST_PATH  = "data/processed/dataset_manifest.csv"
FRAMES_DIR     = "data/processed/frames"
MODEL_SAVE_DIR = "models"
SEQUENCE_LEN   = 16
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
EPOCHS         = 20
BATCH_SIZE     = 8
LEARNING_RATE  = 0.001
FEATURE_DIM    = 512       # ResNet18 output features
CLASSES        = 2         # accident, normal
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

print(f"Using device: {DEVICE}")

# ── Image transforms ─────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

# ── Dataset ──────────────────────────────────────────────
class AccidentSequenceDataset(Dataset):
    def __init__(self, manifest_path, split, frames_dir):
        df = pd.read_csv(manifest_path)
        self.data      = df[df["split"] == split].reset_index(drop=True)
        self.frames_dir = frames_dir
        self.split     = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        label  = int(row["label"])
        cls    = row["class"]
        frames = row["frames"].split("|")

        folder = os.path.join(self.frames_dir, self.split, cls)
        imgs   = []
        for fname in frames:
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                import cv2
                img = cv2.imread(fpath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(transform(img))
            else:
                imgs.append(torch.zeros(3, 224, 224))

        # Stack: (seq_len, C, H, W)
        seq_tensor = torch.stack(imgs)
        return seq_tensor, label

# ── Feature Extractor (ResNet18 backbone, torchvision) ───
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer — use as feature extractor only
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False   # freeze backbone

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        features = self.backbone(x)       # (B*S, 512, 1, 1)
        features = features.view(B, S, -1)  # (B, S, 512)
        return features

# ── LSTM Classifier ──────────────────────────────────────
class AccidentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(
            input_size  = FEATURE_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            batch_first = True,
            dropout     = 0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, CLASSES),
        )

    def forward(self, x):
        features          = self.feature_extractor(x)   # (B, S, 512)
        lstm_out, _       = self.lstm(features)          # (B, S, 256)
        last_hidden       = lstm_out[:, -1, :]           # (B, 256)
        out               = self.classifier(last_hidden) # (B, 2)
        return out

# ── Training loop ─────────────────────────────────────────
def train():
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        print("        Run day2_preprocess.py first.")
        return

    print("\n[1/4] Loading datasets...")
    train_ds = AccidentSequenceDataset(MANIFEST_PATH, "train", FRAMES_DIR)
    val_ds   = AccidentSequenceDataset(MANIFEST_PATH, "val",   FRAMES_DIR)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"  Train sequences: {len(train_ds)}")
    print(f"  Val sequences  : {len(val_ds)}")

    print("\n[2/4] Building model...")
    model     = AccidentLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("\n[3/4] Training LSTM...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Device: {DEVICE}\n")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for seqs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [train]"):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_dl:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                outputs      = model(seqs)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total if train_total else 0
        val_acc   = val_correct   / val_total   if val_total   else 0
        scheduler.step()

        print(f"  Epoch {epoch+1:02d}: "
              f"loss={train_loss/len(train_dl):.4f}  "
              f"train_acc={train_acc:.2%}  "
              f"val_acc={val_acc:.2%}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (val_acc={best_acc:.2%})")

    print(f"\n[4/4] Training complete.")
    print(f"  Best val accuracy : {best_acc:.2%}")
    print(f"  Model saved to    : {MODEL_SAVE_DIR}/lstm_best.pt")

    if best_acc >= 0.80:
        print("  STATUS: PASSED (>=80%) — ready for Day 4")
    else:
        print("  STATUS: BELOW TARGET")
        print("  Tips:")
        print("    1. Increase EPOCHS to 40")
        print("    2. Unfreeze backbone: set param.requires_grad = True")
        print("    3. Add more training videos")

if __name__ == "__main__":
    print("=" * 55)
    print("  Day 3 Part 2 — LSTM Temporal Classifier")
    print("=" * 55)
    train()
    print("\nDay 3 complete. Next: python day4_pipeline.py")