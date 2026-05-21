"""
Data Quality Fix — Run this BEFORE retraining
Does 3 things:
  1. Removes duplicate/near-duplicate frames (same video frames look identical)
  2. Balances accident vs normal class counts
  3. Verifies image quality (removes blurry frames)
Run: python fix_data.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hashlib
import shutil

FRAMES_DIR  = "data/processed/frames"
YOLO_DIR    = "data/yolo"
BLUR_THRESH = 100.0    # frames below this laplacian variance are blurry
KEEP_EVERY  = 3        # keep 1 of every N frames per video (dedup)

# ─────────────────────────────────────────────────────────

def is_blurry(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESH

def deduplicate_and_clean(split, cls):
    folder = Path(FRAMES_DIR) / split / cls
    if not folder.exists():
        return 0, 0

    files = sorted(folder.glob("*.jpg"))
    total  = len(files)
    kept   = 0
    removed = 0

    # Group by video_id (filename format: class_videoid_framenum.jpg)
    videos = {}
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            vid_id = "_".join(parts[1:-1])
        else:
            vid_id = "unknown"
        videos.setdefault(vid_id, []).append(f)

    for vid_id, frames in videos.items():
        for i, frame_path in enumerate(sorted(frames)):
            # Keep only every Nth frame per video
            if i % KEEP_EVERY != 0:
                frame_path.unlink()
                removed += 1
                continue
            # Remove blurry frames
            if is_blurry(frame_path):
                frame_path.unlink()
                removed += 1
                continue
            kept += 1

    return total, removed

def balance_classes():
    """Make accident and normal have same frame count in train split."""
    acc_dir = Path(FRAMES_DIR) / "train" / "accident"
    nor_dir = Path(FRAMES_DIR) / "train" / "normal"

    acc_files = sorted(acc_dir.glob("*.jpg"))
    nor_files = sorted(nor_dir.glob("*.jpg"))

    acc_count = len(acc_files)
    nor_count = len(nor_files)

    print(f"\n  Before balancing:")
    print(f"    accident : {acc_count}")
    print(f"    normal   : {nor_count}")

    # Trim the larger class
    if acc_count > nor_count:
        for f in acc_files[nor_count:]:
            f.unlink()
        print(f"  Trimmed accident to {nor_count}")
    elif nor_count > acc_count:
        for f in nor_files[acc_count:]:
            f.unlink()
        print(f"  Trimmed normal to {acc_count}")
    else:
        print(f"  Already balanced.")

    final = min(acc_count, nor_count)
    print(f"  Final per class: {final}")
    return final

def rebuild_yolo_dir():
    """Rebuild data/yolo from cleaned data/processed/frames."""
    print("\n  Rebuilding data/yolo from cleaned frames...")
    if Path(YOLO_DIR).exists():
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            src = Path(FRAMES_DIR) / split / cls
            dst = Path(YOLO_DIR) / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                files = list(src.glob("*.jpg"))
                for f in tqdm(files, desc=f"  {split}/{cls}", leave=False):
                    shutil.copy(f, dst / f.name)

    # Print final counts
    print("\n  Final dataset counts:")
    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            d = Path(YOLO_DIR) / split / cls
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"    {split}/{cls}: {count} frames")

if __name__ == "__main__":
    print("=" * 55)
    print("  Data Quality Fix")
    print("=" * 55)

    print("\n[1/3] Removing duplicate + blurry frames...")
    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            total, removed = deduplicate_and_clean(split, cls)
            if total > 0:
                kept = total - removed
                print(f"  {split}/{cls}: {total} → {kept} frames "
                      f"(removed {removed}, {removed/total*100:.1f}%)")

    print("\n[2/3] Balancing classes...")
    balance_classes()

    print("\n[3/3] Rebuilding YOLO dataset...")
    rebuild_yolo_dir()

    print("\nDone! Now retrain:")
    print("  python day3_train_yolo.py")