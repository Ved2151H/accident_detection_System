"""
Fix Data Leakage — Run this BEFORE retraining LSTM
The problem: frames from same video are in both train and val
The fix: split by VIDEO, not by frame
Run: python fix_data_leakage.py
Then: python day3_train_lstm.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Config ───────────────────────────────────────────────
PROCESSED_DIR = "data/processed/frames"
YOLO_DIR      = "data/yolo"
TRAIN_RATIO   = 0.80   # 80% of VIDEOS for train
VAL_RATIO     = 0.20   # 20% of VIDEOS for val
RANDOM_SEED   = 42
# ─────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

def get_video_groups(cls_dir):
    """Group frames by video_id from filename: class_videoid_frame.jpg"""
    groups = defaultdict(list)
    for f in Path(cls_dir).glob("*.jpg"):
        parts = f.stem.split("_")
        vid_id = "_".join(parts[1:-1]) if len(parts) >= 3 else f.stem
        groups[vid_id].append(f)
    return dict(groups)

def rebuild_split(cls, label_name):
    # Collect ALL frames for this class across all splits
    all_frames_by_video = defaultdict(list)

    for split in ["train", "val", "test"]:
        src = Path(PROCESSED_DIR) / split / cls
        if not src.exists():
            continue
        groups = get_video_groups(src)
        for vid_id, frames in groups.items():
            all_frames_by_video[vid_id].extend(frames)

    video_ids = list(all_frames_by_video.keys())
    random.shuffle(video_ids)

    n         = len(video_ids)
    n_train   = int(n * TRAIN_RATIO)
    train_vids = set(video_ids[:n_train])
    val_vids   = set(video_ids[n_train:])

    print(f"\n  [{cls}] Total videos : {n}")
    print(f"    Train videos : {len(train_vids)}")
    print(f"    Val videos   : {len(val_vids)}")

    # Clear old yolo dirs for this class
    for split in ["train", "val"]:
        d = Path(YOLO_DIR) / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Copy frames into correct split based on VIDEO id
    train_count = val_count = 0
    for vid_id, frames in all_frames_by_video.items():
        if vid_id in train_vids:
            dst = Path(YOLO_DIR) / "train" / cls
            for f in frames:
                shutil.copy(f, dst / f.name)
            train_count += len(frames)
        else:
            dst = Path(YOLO_DIR) / "val" / cls
            for f in frames:
                shutil.copy(f, dst / f.name)
            val_count += len(frames)

    print(f"    Train frames : {train_count}")
    print(f"    Val frames   : {val_count}")
    return train_count, val_count

def verify_no_leakage():
    """Confirm no video appears in both train and val."""
    print("\n  Verifying no data leakage...")
    for cls in ["accident", "normal"]:
        train_vids = set()
        val_vids   = set()

        for f in (Path(YOLO_DIR) / "train" / cls).glob("*.jpg"):
            parts = f.stem.split("_")
            vid_id = "_".join(parts[1:-1]) if len(parts) >= 3 else f.stem
            train_vids.add(vid_id)

        for f in (Path(YOLO_DIR) / "val" / cls).glob("*.jpg"):
            parts = f.stem.split("_")
            vid_id = "_".join(parts[1:-1]) if len(parts) >= 3 else f.stem
            val_vids.add(vid_id)

        overlap = train_vids & val_vids
        if overlap:
            print(f"  [WARN] {cls}: {len(overlap)} videos still in both splits!")
        else:
            print(f"  [OK]  {cls}: zero overlap between train and val ✓")

if __name__ == "__main__":
    print("=" * 55)
    print("  Fix Data Leakage — Video-level Split")
    print("=" * 55)

    print("\n[1/3] Rebuilding splits by video (not frame)...")
    for cls in ["accident", "normal"]:
        rebuild_split(cls, cls)

    print("\n[2/3] Verifying split integrity...")
    verify_no_leakage()

    print("\n[3/3] Final dataset counts:")
    for split in ["train", "val"]:
        for cls in ["accident", "normal"]:
            d = Path(YOLO_DIR) / split / cls
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split}/{cls}: {count} frames")

    print("\nDone! Now retrain LSTM:")
    print("  python day3_train_lstm.py")