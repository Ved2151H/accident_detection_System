"""
Threshold Tester — Run this on a known accident video
Shows exactly what YOLO and LSTM are outputting
so you can set the right thresholds in day4_pipeline.py
Run: python test_thresholds.py --source "path/to/accident_video.mp4"
"""

import cv2
import torch
import torch.nn as nn
import collections
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms

# ── Config (must match day4_pipeline.py) ─────────────────
YOLO_WEIGHTS  = "runs/classify/models/accident_detector/weights/best.pt"
LSTM_WEIGHTS  = "models/lstm_best.pt"
SEQUENCE_LEN  = 8
FEATURE_DIM   = 512
HIDDEN_SIZE   = 128
NUM_LAYERS    = 1
NUM_CLASSES   = 2
FRAME_SKIP    = 3
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        return self.backbone(x).view(B, S, FEATURE_DIM)

class AccidentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor  = FeatureExtractor()
        self.lstm       = nn.LSTM(FEATURE_DIM, HIDDEN_SIZE,
                                   NUM_LAYERS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x):
        feat   = self.extractor(x)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])

def test(source):
    print("=" * 55)
    print("  Threshold Tester")
    print("=" * 55)

    yolo = YOLO(YOLO_WEIGHTS)
    lstm = AccidentLSTM().to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=DEVICE))
    lstm.eval()

    cap = cv2.VideoCapture(
        int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    buffer    = collections.deque(maxlen=SEQUENCE_LEN)
    frame_idx = 0

    yolo_acc_scores  = []
    lstm_acc_scores  = []
    detections       = 0

    print(f"\nAnalysing: {source}")
    print(f"{'Frame':>7}  {'YOLO_accident':>14}  {'LSTM_accident':>14}  {'Decision':>10}")
    print("-" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        # YOLO
        results  = yolo(frame, verbose=False)
        top_cls  = results[0].probs.top1
        top_conf = results[0].probs.top1conf.item()
        yolo_acc = top_conf if top_cls == 0 else (1 - top_conf)
        yolo_acc_scores.append(yolo_acc)

        # LSTM
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)
        buffer.append(tensor)

        lstm_acc = 0.0
        if len(buffer) == SEQUENCE_LEN:
            seq = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = lstm(seq)
                probs  = torch.softmax(logits, dim=1)[0]
            lstm_acc = probs[0].item()
            lstm_acc_scores.append(lstm_acc)

        decision = "ACCIDENT" if (yolo_acc > 0.5 and lstm_acc > 0.5) else "normal"
        if decision == "ACCIDENT":
            detections += 1

        if frame_idx % 30 == 0:   # print every 30 processed frames
            print(f"{frame_idx:>7}  {yolo_acc:>14.2%}  {lstm_acc:>14.2%}  {decision:>10}")

    cap.release()

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    if yolo_acc_scores:
        print(f"  YOLO accident prob — min: {min(yolo_acc_scores):.2%}  "
              f"max: {max(yolo_acc_scores):.2%}  "
              f"avg: {sum(yolo_acc_scores)/len(yolo_acc_scores):.2%}")
    if lstm_acc_scores:
        print(f"  LSTM accident prob — min: {min(lstm_acc_scores):.2%}  "
              f"max: {max(lstm_acc_scores):.2%}  "
              f"avg: {sum(lstm_acc_scores)/len(lstm_acc_scores):.2%}")
    print(f"  Frames analysed    : {frame_idx // FRAME_SKIP}")
    print(f"  Detections (>50%)  : {detections}")

    print("\n  ── Recommended thresholds for day4_pipeline.py ──")
    if yolo_acc_scores and lstm_acc_scores:
        rec_yolo = max(0.40, min(yolo_acc_scores) - 0.05)
        rec_lstm = max(0.40, min(lstm_acc_scores) - 0.05)
        print(f"  ACCIDENT_CONF  = {rec_yolo:.2f}")
        print(f"  LSTM_THRESHOLD = {rec_lstm:.2f}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Path to a known accident video")
    args = parser.parse_args()
    test(args.source)