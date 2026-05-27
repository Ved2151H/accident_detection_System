import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

DIRS_TO_CLEAN = [
    "data/yolo",
    "data/processed/frames"
]

def check_image(fp):
    fp_str = str(fp)
    # Check 1: 0-byte file (very fast)
    try:
        size = os.path.getsize(fp_str)
        if size == 0:
            return fp_str, "0-byte file"
    except Exception as e:
        return fp_str, f"Size error: {e}"
        
    # Check 2: Try verifying the image header (extremely fast, doesn't decode pixels)
    try:
        with Image.open(fp_str) as img:
            img.verify()
    except Exception as e:
        return fp_str, f"Corrupt image: {e}"
        
    return fp_str, None

def scan_and_clean():
    print("=" * 55)
    print("  AEGIS EYE DATASET CLEANER - MULTI-THREADED SCANNER")
    print("=" * 55)
    
    corrupt_deleted = 0
    
    for base_dir in DIRS_TO_CLEAN:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Directory not found: {base_dir} (skipping)")
            continue
            
        print(f"\nScanning: {base_dir} ...")
        jpg_files = list(base_path.glob("**/*.jpg"))
        total_files = len(jpg_files)
        print(f"Found {total_files} images to verify.")
        
        # Parallel scan using 32 worker threads
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            # We wrap in tqdm to see progress of scheduling
            futures = [executor.submit(check_image, fp) for fp in jpg_files]
            for future in tqdm(futures, desc="Scanning"):
                results.append(future.result())
                
        # Handle corrupt files
        print("Processing scan results...")
        for fp_str, error_msg in results:
            if error_msg is not None:
                corrupt_deleted += 1
                print(f"  [CORRUPT] Deleting: {fp_str} ({error_msg})")
                try:
                    os.remove(fp_str)
                except Exception as e:
                    print(f"    [ERROR] Failed to delete: {e}")
                    
    print("\n" + "=" * 55)
    print(f"  CLEANUP COMPLETE")
    print(f"  Total corrupt deleted: {corrupt_deleted}")
    print("=" * 55)

if __name__ == "__main__":
    scan_and_clean()
