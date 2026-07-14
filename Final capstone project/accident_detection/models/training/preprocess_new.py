import os
import cv2
import shutil
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurations
DATASET_PATH = r"D:\Subjects_Languages\Languages\VED-DEVANAND-DHANOKAR-g37-ai-ml\Final capstone project\Real_dataset_accident"
CSV_PATH = os.path.join(DATASET_PATH, "metadata-real.csv")
OUTPUT_DIR = Path("data/processed/frames")
YOLO_DIR = Path("data/yolo")
IMG_SIZE = (224, 224)
MAX_FRAMES_PER_CLASS = 16  # Extract up to 16 frames for normal and 16 for accident per video

def clean_and_prepare_dirs():
    print("[1/4] Cleaning existing processed and YOLO directories...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)
            os.makedirs(YOLO_DIR / split / cls, exist_ok=True)

def split_dataset(df):
    print("[2/4] Splitting dataset into train (70%), val (15%), test (15%)...")
    # Set seed for reproducibility
    random.seed(42)
    
    video_paths = df['path'].tolist()
    random.shuffle(video_paths)
    
    n = len(video_paths)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    
    train_vids = set(video_paths[:train_end])
    val_vids = set(video_paths[train_end:val_end])
    test_vids = set(video_paths[val_end:])
    
    splits = {}
    for p in video_paths:
        if p in train_vids:
            splits[p] = "train"
        elif p in val_vids:
            splits[p] = "val"
        else:
            splits[p] = "test"
            
    print(f"  Splits summary: {len(train_vids)} train, {len(val_vids)} val, {len(test_vids)} test")
    return splits

def process_single_video(task_args):
    """Processes a single video: extracts frames and saves them. Runs in parallel."""
    video_rel_path, accident_frame, no_frames, split, dataset_path, output_dir, img_size, max_frames = task_args
    video_abs_path = os.path.join(dataset_path, video_rel_path)
    video_id = Path(video_rel_path).stem
    
    # Calculate target frame indices to save
    normal_start = max(0, accident_frame - max_frames)
    normal_end = accident_frame
    normal_indices = set(range(normal_start, normal_end))
    
    accident_start = accident_frame
    accident_end = min(no_frames, accident_frame + max_frames)
    accident_indices = set(range(accident_start, accident_end))
    
    max_target_idx = max(accident_end, normal_end)
    
    cap = cv2.VideoCapture(video_abs_path)
    if not cap.isOpened():
        return False
        
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_target_idx:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in normal_indices:
            resized = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
            out_path = Path(output_dir) / split / "normal" / f"normal_{video_id}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), resized)
            
        elif frame_idx in accident_indices:
            resized = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
            out_path = Path(output_dir) / split / "accident" / f"accident_{video_id}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), resized)
            
        frame_idx += 1
        
    cap.release()
    return True

def extract_frames_parallel(df, splits):
    print("[3/4] Extracting frames from videos in parallel...")
    
    # Build list of tasks
    tasks = []
    for idx, row in df.iterrows():
        video_rel_path = row['path']
        accident_frame = int(row['accident_frame'])
        no_frames = int(row['no_frames'])
        split = splits[video_rel_path]
        
        tasks.append((
            video_rel_path,
            accident_frame,
            no_frames,
            split,
            DATASET_PATH,
            str(OUTPUT_DIR),
            IMG_SIZE,
            MAX_FRAMES_PER_CLASS
        ))
        
    # We can use ProcessPoolExecutor to use CPU cores
    # Use 12 workers for RTX 4060 / i7-14700HX (20 cores) to keep it extremely fast but not starve system
    max_workers = min(12, os.cpu_count() or 4)
    print(f"  Using ProcessPoolExecutor with {max_workers} workers.")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            if future.result():
                success_count += 1
                
    print(f"  Processed successfully: {success_count} out of {len(tasks)} videos.")

def copy_to_yolo_dir():
    print("[4/4] Copying frames to YOLO directory structure...")
    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            src_dir = OUTPUT_DIR / split / cls
            dst_dir = YOLO_DIR / split / cls
            
            files = list(src_dir.glob("*.jpg"))
            for f in tqdm(files, desc=f"Copying {split}/{cls}", leave=False):
                shutil.copy(f, dst_dir / f.name)
                
    # Print counts
    for split in ["train", "val", "test"]:
        for cls in ["accident", "normal"]:
            count = len(list((YOLO_DIR / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {count} images")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV path does not exist: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    clean_and_prepare_dirs()
    splits = split_dataset(df)
    extract_frames_parallel(df, splits)
    copy_to_yolo_dir()
    print("[SUCCESS] Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
