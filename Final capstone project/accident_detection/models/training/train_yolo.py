"""
Day 3 - Part 1: YOLOv8 Classification Training
Fine-tunes YOLOv8n-cls on processed frames
Labels: accident(0), normal(1)
Optimized for RTX 4060 + i7-14700HX
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
PROCESSED_DIR  = "data/processed/frames"
YOLO_DATA_DIR  = "data/yolo"
MODEL_SAVE_DIR = "models"
EPOCHS         = 50
IMG_SIZE       = 224
BATCH_SIZE     = 32       # Safer for 5GB VRAM limit
WORKERS        = 8        # increased to accelerate data loading/caching
PRETRAINED     = "yolov8s-cls.pt"   # MUST be cls version for classification
DEVICE         = 0                  # 0 = GPU, "cpu" = CPU


def convert_to_yolo_format():
    """Copy frames into YOLO classify folder structure."""
    print("[1/3] Converting frames to YOLO classify format...")

    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            src = os.path.join(PROCESSED_DIR, split, cls)
            dst = os.path.join(YOLO_DATA_DIR, split, cls)
            os.makedirs(dst, exist_ok=True)

            if not os.path.exists(src):
                print(f"  [SKIP] {src} not found")
                continue

            files = list(Path(src).glob("*.jpg"))
            if not files:
                print(f"  [SKIP] No .jpg files in {src}")
                continue

            for f in tqdm(files, desc=f"  Copying {split}/{cls}"):
                dest_file = os.path.join(dst, f.name)
                if not os.path.exists(dest_file):
                    shutil.copy(f, dest_file)

    # Print summary
    for split in ["train", "val", "test"]:
        total = sum(
            len(list(Path(os.path.join(YOLO_DATA_DIR, split, c)).glob("*.jpg")))
            for c in ["accident", "normal"]
            if os.path.exists(os.path.join(YOLO_DATA_DIR, split, c))
        )
        print(f"  {split}: {total} images")

    print(f"  Done. YOLO dataset at: {YOLO_DATA_DIR}\n")


def train_yolo_classifier():
    """Fine-tune YOLOv8n-cls on RTX 4060."""
    from ultralytics import YOLO

    print("[2/3] Starting YOLOv8 classification training on GPU...")
    print(f"  Model      : {PRETRAINED}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Workers    : {WORKERS}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Device     : GPU (CUDA:{DEVICE})\n")

    model = YOLO(PRETRAINED)

    results = model.train(
        task          = "classify",
        data          = YOLO_DATA_DIR,
        epochs        = EPOCHS,
        imgsz         = IMG_SIZE,
        batch         = BATCH_SIZE,
        workers       = WORKERS,
        cache         = True,
        device        = DEVICE,
        name          = "accident_detector",
        project       = MODEL_SAVE_DIR,
        patience      = 50,
        optimizer     = "AdamW",
        lr0           = 0.0002,
        dropout       = 0.0,
        augment       = False,
        weight_decay  = 0.0,
        warmup_epochs = 2,
        plots         = True,
        save          = True,
        verbose       = True,
        exist_ok      = True,    # overwrite previous run folder
    )

    best = os.path.join("runs", "classify", MODEL_SAVE_DIR, "accident_detector", "weights", "best.pt")
    print(f"\n  Training complete.")
    print(f"  Best weights: {best}")
    return results


def evaluate_model():
    """Run validation on the best saved model."""
    from ultralytics import YOLO

    print("\n[3/3] Evaluating best model on validation set...")

    best_weights = os.path.join("runs", "classify", MODEL_SAVE_DIR, "accident_detector", "weights", "best.pt")
    if not os.path.exists(best_weights):
        print("  [ERROR] best.pt not found — training may have failed.")
        return

    model   = YOLO(best_weights)
    metrics = model.val(
        data   = YOLO_DATA_DIR,
        split  = "val",
        device = DEVICE,
    )

    print(f"\n--- Validation Results ---")
    print(f"  Top-1 Accuracy : {metrics.top1:.2%}")
    print(f"  Top-5 Accuracy : {metrics.top5:.2%}")

    if metrics.top1 >= 0.80:
        print("  STATUS: PASSED (>=80%) - ready for LSTM training")
    else:
        print("  STATUS: BELOW TARGET")
        print("  Tips to improve:")
        print("    1. Increase EPOCHS to 50")
        print("    2. Lower lr0 to 0.0005")
        print("    3. Remove blurry or mislabeled frames from data/")
        print("    4. Try yolov8s-cls.pt (small) for better accuracy")


if __name__ == "__main__":
    import torch

    # ── GPU Check ────────────────────────────────────────
    print("=" * 55)
    print("  GPU / Device Check")
    print("=" * 55)
    if torch.cuda.is_available():
        print(f"  [OK]  CUDA available")
        print(f"  [OK]  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  [OK]  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Set memory limit: 5GB of 8GB is 0.625 fraction
        torch.cuda.set_per_process_memory_fraction(0.625, 0)
        print("  [OK]  VRAM limit set to 5GB (0.625 fraction)")
    else:
        print("  [WARN] CUDA not found — running on CPU")
        print("  Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        DEVICE = "cpu"
    print()

    print("=" * 55)
    print("  Day 3 Part 1 — YOLOv8 Classifier Training")
    print("=" * 55)

    convert_to_yolo_format()
    train_yolo_classifier()
    evaluate_model()

    print("\nPart 1 done. Now run: python day3_train_lstm.py")
