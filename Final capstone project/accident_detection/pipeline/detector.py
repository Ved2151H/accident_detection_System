"""
Day 4 — Live Inference Pipeline
Reads video (file or CCTV stream) → YOLO classify → LSTM temporal →
triggers alert on accident detection → logs to SQLite
Run: python day4_pipeline.py --source video.mp4
     python day4_pipeline.py --source 0        (webcam)
     python day4_pipeline.py --source rtsp://...  (IP camera)
"""

import os
import cv2
import time
import sqlite3
import argparse
import datetime
import threading
import collections
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────
YOLO_WEIGHTS   = "runs/classify/models/accident_detector/weights/best.pt"
LSTM_WEIGHTS   = "models/lstm_best.pt"
DB_PATH        = "logs/incidents.db"
SNAPSHOTS_DIR  = "logs/snapshots"
SEQUENCE_LEN   = 8
FEATURE_DIM    = 512
HIDDEN_SIZE    = 128
NUM_LAYERS     = 1
NUM_CLASSES    = 2
FRAME_SKIP     = 3        # process every Nth frame (speed vs accuracy)
ACCIDENT_CONF  = 0.30     # YOLO confidence threshold to flag accident
LSTM_THRESHOLD = 0.48     # LSTM probability threshold to confirm accident
COOLDOWN_SECS  = 10       # min seconds between two alerts (avoid spam)
DISPLAY        = True     # show live window (set False for headless server)
# ─────────────────────────────────────────────────────────

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Transform ────────────────────────────────────────────
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── LSTM Model (must match day3_train_lstm.py) ───────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        B, S, C, H, W = x.shape
        x    = x.view(B * S, C, H, W)
        feat = self.backbone(x)
        return feat.view(B, S, FEATURE_DIM)

class AccidentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor  = FeatureExtractor()
        self.lstm       = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE,
                                   NUM_LAYERS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        feat     = self.extractor(x)
        out, _   = self.lstm(feat)
        return self.classifier(out[:, -1, :])

# ── Database ─────────────────────────────────────────────
def init_db():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            source      TEXT,
            yolo_conf   REAL,
            lstm_prob   REAL,
            snapshot    TEXT,
            latitude    REAL,
            longitude   REAL
        )
    """)
    # Database migration checks
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(incidents)")
        columns = [row[1] for row in cursor.fetchall()]
        if "latitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN latitude REAL")
        if "longitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN longitude REAL")
    except Exception as e:
        print(f"DB Migration Error: {e}")
    conn.commit()
    conn.close()

def log_incident(source, yolo_conf, lstm_prob, snapshot_path, latitude=18.5204, longitude=73.8567):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO incidents (timestamp, source, yolo_conf, lstm_prob, snapshot, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.now().isoformat(), source,
          yolo_conf, lstm_prob, snapshot_path, latitude, longitude))
    conn.commit()
    conn.close()

# ── Alert ─────────────────────────────────────────────────
def send_alert(timestamp, yolo_conf, lstm_prob, snapshot_path, source):
    """Alert in terminal + optional email (Day 7 adds full email)."""
    print("\n" + "!" * 55)
    print(f"  🚨 ACCIDENT DETECTED")
    print(f"  Time     : {timestamp}")
    print(f"  Source   : {source}")
    print(f"  YOLO     : {yolo_conf:.2%}")
    print(f"  LSTM     : {lstm_prob:.2%}")
    print(f"  Snapshot : {snapshot_path}")
    print("!" * 55 + "\n")

    # Email alert stub — will be completed in Day 7
    # from alerts.email_alert import send_email
    # send_email(timestamp, snapshot_path, lstm_prob)

# ── Main Pipeline ─────────────────────────────────────────
class AccidentDetector:
    def __init__(self, source):
        self.source      = source
        self.last_alert  = 0   # timestamp of last alert

        print("=" * 55)
        print("  Accident Detection Pipeline — Day 4")
        print("=" * 55)

        # Load YOLO
        print(f"\n[1/3] Loading YOLO from {YOLO_WEIGHTS}...")
        if not Path(YOLO_WEIGHTS).exists():
            raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
        self.yolo = YOLO(YOLO_WEIGHTS)
        print("  YOLO loaded ✓")

        # Load LSTM
        print(f"\n[2/3] Loading LSTM from {LSTM_WEIGHTS}...")
        if not Path(LSTM_WEIGHTS).exists():
            raise FileNotFoundError(f"LSTM weights not found: {LSTM_WEIGHTS}")
        self.lstm = AccidentLSTM().to(DEVICE)
        self.lstm.load_state_dict(torch.load(LSTM_WEIGHTS,
                                              map_location=DEVICE))
        self.lstm.eval()
        print("  LSTM loaded ✓")

        # Init DB
        print(f"\n[3/3] Initialising database at {DB_PATH}...")
        init_db()
        print("  Database ready ✓")

        # Frame buffer for LSTM sequences
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LEN)

        print(f"\n  Device     : {DEVICE}")
        print(f"  Frame skip : every {FRAME_SKIP} frames")
        print(f"  YOLO thr   : {ACCIDENT_CONF:.0%}")
        print(f"  LSTM thr   : {LSTM_THRESHOLD:.0%}")
        print(f"  Cooldown   : {COOLDOWN_SECS}s between alerts")
        print(f"\nStarting stream: {source}\n")

    def preprocess_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_transform(rgb)

    def run_lstm(self):
        """Run LSTM on current frame buffer. Returns (accident_prob, normal_prob)."""
        if len(self.frame_buffer) < SEQUENCE_LEN:
            return None

        seq    = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                logits = self.lstm(seq)
                probs  = torch.softmax(logits, dim=1)[0]
        # label 0 = accident, label 1 = normal
        return probs[0].item(), probs[1].item()

    def save_snapshot(self, frame):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOTS_DIR, f"incident_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def draw_overlay(self, frame, status, yolo_conf, lstm_prob):
        h, w = frame.shape[:2]

        # Status bar background
        color = (0, 0, 200) if status == "ACCIDENT" else (0, 180, 0)
        cv2.rectangle(frame, (0, 0), (w, 50), color, -1)

        # Status text
        text = f"STATUS: {status}"
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Confidence scores
        info = f"YOLO: {yolo_conf:.0%}  LSTM: {lstm_prob:.0%}"
        cv2.putText(frame, info, (w - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        # Open video source
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_idx  = 0
        yolo_conf  = 0.0
        lstm_prob  = 0.0
        status     = "NORMAL"

        print("Running... Press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            frame_idx += 1

            # Process every Nth frame
            if frame_idx % FRAME_SKIP == 0:
                tensor = self.preprocess_frame(frame)
                self.frame_buffer.append(tensor)

                # ── YOLO classification ──
                results   = self.yolo(frame, verbose=False)
                top_cls   = results[0].probs.top1
                top_conf  = results[0].probs.top1conf.item()

                yolo_accident = (top_cls == 0 and top_conf >= ACCIDENT_CONF)
                yolo_conf     = top_conf if top_cls == 0 else (1 - top_conf)

                # ── LSTM temporal check ──
                lstm_result = self.run_lstm()
                if lstm_result:
                    lstm_prob = lstm_result[0]  # accident probability

                    # ── Combined decision ──
                    now = time.time()
                    cooldown_ok = (now - self.last_alert) >= COOLDOWN_SECS

                    if (yolo_accident and
                        lstm_prob >= LSTM_THRESHOLD and
                        cooldown_ok):

                        status          = "ACCIDENT"
                        self.last_alert = now
                        snapshot        = self.save_snapshot(frame)
                        ts              = datetime.datetime.now().isoformat()

                        log_incident(str(self.source), yolo_conf,
                                     lstm_prob, snapshot)
                        # Run alert in background thread
                        threading.Thread(
                            target=send_alert,
                            args=(ts, yolo_conf, lstm_prob,
                                  snapshot, self.source),
                            daemon=True
                        ).start()
                    else:
                        status = "NORMAL"

            # ── Display ──
            if DISPLAY:
                display_frame = self.draw_overlay(
                    frame.copy(), status, yolo_conf, lstm_prob)
                cv2.imshow("Accident Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit by user.")
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\nPipeline stopped.")
        print(f"Incidents logged to: {DB_PATH}")
        print(f"Snapshots saved in : {SNAPSHOTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accident Detection Pipeline")
    parser.add_argument(
        "--source", default="0",
        help="Video source: file path, 0 for webcam, rtsp:// for IP cam")
    args = parser.parse_args()

    detector = AccidentDetector(args.source)
    detector.run()