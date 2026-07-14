"""
Helmet Detection Utilities.
Includes automatic model downloading, CSV violation logging,
evidence snapshot saving, and future expansion stubs.
"""

import os
import cv2
import csv
import datetime
import urllib.request

def download_helmet_model(target_path="models/helmet_detection/weights/helmet_model.pt"):
    """
    Downloads the pre-trained YOLOv8 helmet detection model from Hugging Face
    if it does not exist locally.
    """
    if os.path.exists(target_path):
        return target_path

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Hugging Face resolve URL for iam-tsr/yolov8n-helmet-detection best.pt weights
    url = "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt"
    
    print(f"Downloading helmet model from {url}...")
    try:
        # Request with a standard User-Agent to prevent 403 Forbidden errors
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Model saved successfully to {target_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Try a fallback download URL or raise
        raise RuntimeError(f"Could not download helmet detection model: {e}")
        
    return target_path


def log_violation_to_csv(track_id, status, snapshot_path, csv_path="logs/helmet_violations.csv"):
    """
    Logs a helmet violation event to a CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header
                writer.writerow([
                    "Timestamp", 
                    "Motorcycle Track ID", 
                    "Violation Type", 
                    "Snapshot Path", 
                    "License Plate (ANPR Stub)", 
                    "Challan Status (Stub)"
                ])
            writer.writerow([
                timestamp, 
                track_id, 
                status, 
                snapshot_path, 
                "PENDING (Future ANPR)", 
                "DRAFTED (Future Challan)"
            ])
    except Exception as e:
        print(f"Failed to write to CSV log: {e}")


def save_evidence_image(frame, track_id, output_dir="logs/snapshots/helmet_violations"):
    """
    Saves the current frame showing a violation to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_bike_{track_id}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Save frame in BGR format
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        print(f"Failed to save evidence image: {e}")
        return ""


# ── Future Expansion Architectural Stubs ──────────────────────────────

def trigger_anpr_future_stub(frame, bike_box):
    """
    Architectural Slot for ANPR (Automatic Number Plate Recognition).
    This function will be responsible for:
    1. Cropping the motorcycle license plate area (bottom-rear of the bike box).
    2. Passing the cropped frame to an OCR engine (e.g. EasyOCR, Tesseract, or a fine-tuned YOLO plate detector).
    3. Returning the license plate text.
    """
    # Placeholder implementation
    x1, y1, x2, y2 = map(int, bike_box)
    # 1. crop plate region (usually bottom portion of the bike box)
    # crop = frame[int(y1 + (y2-y1)*0.6):y2, x1:x2]
    # 2. run OCR model
    # plate_text = ocr_model(crop)
    return "ANPR_STUB_PLATE_1234"


def generate_challan_future_stub(license_plate, violation_type, snapshot_path):
    """
    Architectural Slot for Automated Challan / Ticket Generation.
    This function will be responsible for:
    1. Connecting to the vehicle registry API using the extracted license plate.
    2. Constructing an electronic challan record with the violation type and snapshot URL.
    3. Posting the data to the traffic authority database/API.
    4. Sending an SMS/email alert to the violator.
    """
    # Placeholder implementation
    # record = {
    #     "license_plate": license_plate,
    #     "violation": violation_type,
    #     "snapshot": snapshot_path,
    #     "fine_amount": 1000, # e.g. 1000 INR
    #     "timestamp": datetime.datetime.now().isoformat()
    # }
    # api.post("/challan", json=record)
    return True
