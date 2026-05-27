"""
Day 2 - Data Preprocessing Script
Targets: RoadAccidents + Normal only
Works with ANY folder of videos you collect yourself
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import random

# ── Config ──────────────────────────────────────────────
RAW_DATA_DIR   = r"D:\Subjects_Languages\Languages\VED-DEVANAND-DHANOKAR-g37-ai-ml\Final capstone project\accident_detection\data\raw"          # put your videos here
OUTPUT_DIR     = r"D:\Subjects_Languages\Languages\VED-DEVANAND-DHANOKAR-g37-ai-ml\Final capstone project\accident_detection\data\processed"
FRAME_SIZE     = (640, 640)          # YOLOv8 default input
FRAME_SKIP     = 5                   # extract every 5th frame
SEQUENCE_LEN   = 16                  # frames per LSTM sequence
TRAIN_RATIO    = 0.95
VAL_RATIO      = 0.05
# TEST_RATIO   = 0.15 (remainder)

# ── Expected folder structure inside data/raw/ ───────────
# data/raw/
#   accident/        ← your accident videos (.mp4, .avi)
#   normal/          ← normal road / street videos

CLASSES = {
    "accident": 0,
    "normal":   1,
}

# ────────────────────────────────────────────────────────

def extract_frames(video_path, output_folder, label, video_id):
    """Extract frames from a single video at FRAME_SKIP interval."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            frame = cv2.resize(frame, FRAME_SIZE)
            fname = f"{label}_{video_id}_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved

def build_sequences(frames_dir, label_id, video_id):
    """
    Group extracted frames into sequences of SEQUENCE_LEN for LSTM.
    Returns list of (sequence_path_list, label) tuples.
    """
    frames = sorted([
        f for f in os.listdir(frames_dir)
        if f.startswith(f"{list(CLASSES.keys())[label_id]}_{video_id}")
    ])
    sequences = []
    for i in range(0, len(frames) - SEQUENCE_LEN + 1, SEQUENCE_LEN):
        seq = frames[i:i + SEQUENCE_LEN]
        if len(seq) == SEQUENCE_LEN:
            sequences.append((seq, label_id))
    return sequences

def split_videos(video_list):
    """Split video list into train/val/test."""
    random.shuffle(video_list)
    n = len(video_list)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)
    return video_list[:train_end], video_list[train_end:val_end], video_list[val_end:]

def process_all():
    print("=" * 55)
    print("  Day 2 — Preprocessing: Accident Detection — Binary Classification")
    print("=" * 55)

    all_records = []
    frames_dir  = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for class_name, label_id in CLASSES.items():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"\n[WARNING] Folder not found: {class_dir}")
            print(f"          Create it and add your {class_name} videos.")
            continue

        videos = list(Path(class_dir).glob("*.mp4")) + \
                 list(Path(class_dir).glob("*.avi")) + \
                 list(Path(class_dir).glob("*.mov"))

        if not videos:
            print(f"\n[WARNING] No videos found in {class_dir}")
            continue

        print(f"\n[{class_name.upper()}] Found {len(videos)} videos")
        train_vids, val_vids, test_vids = split_videos(videos)

        for split_name, split_vids in [("train", train_vids),
                                        ("val",   val_vids),
                                        ("test",  test_vids)]:
            out_dir = os.path.join(frames_dir, split_name, class_name)
            os.makedirs(out_dir, exist_ok=True)

            for vid_path in tqdm(split_vids, desc=f"  {split_name}"):
                vid_id = vid_path.stem
                n_saved = extract_frames(vid_path, out_dir, class_name, vid_id)
                seqs    = build_sequences(out_dir, label_id, vid_id)

                for seq, lbl in seqs:
                    all_records.append({
                        "split":      split_name,
                        "class":      class_name,
                        "label":      lbl,
                        "video_id":   vid_id,
                        "frames":     "|".join(seq),
                        "seq_length": len(seq),
                    })

                print(f"    {vid_path.name}: {n_saved} frames, {len(seqs)} sequences")

    # Save manifest CSV
    df = pd.DataFrame(all_records)
    manifest_path = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")
    df.to_csv(manifest_path, index=False)

    print("\n-- Dataset Summary ----------------------")
    if not df.empty:
        print(df.groupby(["split", "class"])["seq_length"].count().rename("sequences"))
        print(f"\nTotal sequences : {len(df)}")
        print(f"Manifest saved  : {manifest_path}")
    else:
        print("No data processed yet — add videos to data/raw/ first.")

    print("\nDay 2 preprocessing complete.")
    print("Next: Run day3_train.py to augment training data.")

if __name__ == "__main__":
    process_all()