"""
Day 3 - Part 2: LSTM Temporal Anomaly Classifier (Fixed)
Directly reads frames from data/yolo/train|val folders
No manifest dependency — simpler and more reliable
Backbone: ResNet18 (torchvision, frozen)
Optimized for RTX 4060 + i7-14700HX
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# ── Config ───────────────────────────────────────────────
YOLO_DIR       = "data/yolo"       # reads directly from yolo folder
MODEL_SAVE_DIR = "models"
SEQUENCE_LEN   = 8                 # reduced from 16 — more sequences available
HIDDEN_SIZE    = 64               # Match dashboard architecture
NUM_LAYERS     = 1                 # reduced — simpler model
EPOCHS         = 80
BATCH_SIZE     = 32
NUM_WORKERS    = 0
LEARNING_RATE  = 0.0005
FEATURE_DIM    = 512               # ResNet18 output
NUM_CLASSES    = 2                 # 0=accident, 1=normal

# ── Device ───────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ── Transforms ───────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Builds sequences directly from yolo folder structure.
    Groups frames by video_id from filename: class_videoid_frameN.jpg
    Falls back to random sampling if grouping fails.
    """
    def __init__(self, split, transform):
        self.transform  = transform
        self.sequences  = []   # list of (frame_path_list, label)
        self.split      = split
        self.cache      = {}   # path -> 112x112 BGR numpy array

        classes = {"accident": 0, "normal": 1}

        for cls, label in classes.items():
            cls_dir = Path(YOLO_DIR) / split / cls
            if not cls_dir.exists():
                print(f"  [WARN] Not found: {cls_dir}")
                continue

            all_frames = sorted(cls_dir.glob("*.jpg"))
            if not all_frames:
                print(f"  [WARN] No frames in {cls_dir}")
                continue

            # Group frames by video_id
            video_groups = {}
            for f in all_frames:
                parts = f.stem.split("_")
                # filename: accident_videoname_00000
                vid_id = "_".join(parts[1:-1]) if len(parts) >= 3 else "default"
                video_groups.setdefault(vid_id, []).append(f)

            # Build sequences from each video group
            seq_count = 0
            for vid_id, frames in video_groups.items():
                frames = sorted(frames)
                # Use a larger stride to avoid overlapping sequence redundancy
                if self.split == "train":
                    stride = SEQUENCE_LEN * 2
                else:
                    stride = SEQUENCE_LEN
                for i in range(0, len(frames) - SEQUENCE_LEN + 1, stride):
                    seq = frames[i : i + SEQUENCE_LEN]
                    if len(seq) == SEQUENCE_LEN:
                        self.sequences.append((seq, label))
                        seq_count += 1

            print(f"  {split}/{cls}: {len(all_frames)} frames -> "
                  f"{seq_count} sequences")

        random.shuffle(self.sequences)
        print(f"  Total {split} sequences: {len(self.sequences)}\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        imgs = []
        for fp in frame_paths:
            fp_str = str(fp)
            if fp_str in self.cache:
                img_resized = self.cache[fp_str]
            else:
                img = cv2.imread(fp_str)
                if img is None:
                    img_resized = np.zeros((112, 112, 3), dtype=np.uint8)
                else:
                    if img.shape[0] != 112 or img.shape[1] != 112:
                        img_resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
                    else:
                        img_resized = img
                self.cache[fp_str] = img_resized
            
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            imgs.append(self.transform(img_rgb))
        return torch.stack(imgs), label   # (seq_len, C, H, W), label

# ── Feature Extractor ────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Unfreeze ResNet18 backbone parameters to fine-tune and overfit
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        B, S, C, H, W = x.shape
        x        = x.view(B * S, C, H, W)
        feat     = self.backbone(x)            # (B*S, 512, 1, 1)
        feat     = feat.view(B, S, FEATURE_DIM)  # (B, S, 512)
        return feat

# ── Attentive Accident LSTM Model ────────────────────────
class AccidentLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.extractor  = FeatureExtractor()
        self.lstm       = nn.LSTM(input_dim, hidden_size,
                                   num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 64), nn.ReLU(),
            nn.Dropout(0.0), nn.Linear(64, num_classes),
        )
    def forward(self, x):
        feat   = self.extractor(x)             # (B, S, 512)
        out, _ = self.lstm(feat)               # (B, S, 64)
        last   = out[:, -1, :]                  # (B, 64)
        mean_p = out.mean(dim=1)               # (B, 64)
        max_p  = out.max(dim=1)[0]             # (B, 64)
        combined = torch.cat([last, mean_p, max_p], dim=1) # (B, 192)
        return self.classifier(combined)

# ── Train / Eval helpers ─────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for seqs, labels in tqdm(loader,
                                  desc="  train" if training else "  val  ",
                                  leave=False):
            seqs   = seqs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                outputs = model(seqs)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total if total else 0

# ── Main ─────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  GPU / Device Check")
    print("=" * 55)
    if torch.cuda.is_available():
        print(f"  [OK]  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  [OK]  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        # Set VRAM limit: 5GB is 0.625 fraction of 8GB
        torch.cuda.set_per_process_memory_fraction(0.625, 0)
        print("  [OK]  VRAM limit set to 5GB (0.625 fraction)")
    else:
        print("  [WARN] Running on CPU - will be slow")
    print()

    print("=" * 55)
    print("  Day 3 Part 2 - LSTM Temporal Classifier")
    print("=" * 55)

    print("\n[1/4] Building datasets...")
    train_ds = SequenceDataset("train", train_transform)
    val_ds   = SequenceDataset("val",   val_transform)

    if len(train_ds) == 0:
        print("[ERROR] No training sequences found.")
        print("        Make sure data/yolo/train/accident and")
        print("        data/yolo/train/normal folders have .jpg files.")
        return

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n[2/4] Building model...")
    model     = AccidentLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_acc = 0.0

    print(f"  Sequences  : {len(train_ds)} train / {len(val_ds)} val")
    print(f"  Seq length : {SEQUENCE_LEN} frames")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Device     : {DEVICE}")
    print()

    print("[3/4] Training...\n")
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_dl, criterion, optimizer, scaler, training=True)
        val_loss,   val_acc   = run_epoch(
            model, val_dl,   criterion, optimizer, scaler, training=False)
        scheduler.step()

        improved = "[NEW BEST]" if val_acc > best_acc else " "
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"loss={train_loss:.4f} | "
              f"train={train_acc:.2%} | "
              f"val={val_acc:.2%} {improved}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_SAVE_DIR, "lstm_best.pt"))
        else:
            no_improve += 1
            if no_improve >= 30:
                print(f"\n  Early stopping - no improvement for 30 epochs.")
                break

    print(f"\n[4/4] Done.")
    print(f"  Best val accuracy : {best_acc:.2%}")
    print(f"  Saved to          : {MODEL_SAVE_DIR}/lstm_best.pt")

    if best_acc >= 0.80:
        print("  STATUS: PASSED (>=80%) - ready for Day 4 [OK]")
    elif best_acc >= 0.70:
        print("  STATUS: ACCEPTABLE (70-80%) - can proceed to Day 4")
        print("  Note: Add more varied videos later to push above 80%")
    else:
        print("  STATUS: BELOW TARGET")
        print("  Try: unfreeze backbone by removing 'param.requires_grad = False'")

if __name__ == "__main__":
    train()