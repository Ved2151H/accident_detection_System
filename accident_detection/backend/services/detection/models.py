import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from backend.utils.config import (
    YOLO_WEIGHTS, LSTM_WEIGHTS, get_device, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES
)

# ── Transform ────────────────────────────────────────────
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── LSTM Model Definitions ───────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        return self.backbone(x).view(B, S, FEATURE_DIM)

class AttentiveAccidentLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.extractor  = FeatureExtractor()
        self.lstm       = nn.LSTM(input_dim, hidden_size,
                                   num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes),
        )
    def forward(self, x):
        feat   = self.extractor(x)
        out, _ = self.lstm(feat)
        last   = out[:, -1, :]
        mean_p = out.mean(dim=1)
        max_p  = out.max(dim=1)[0]
        combined = torch.cat([last, mean_p, max_p], dim=1)
        return self.classifier(combined)

# Alias for compatibility
AccidentLSTM = AttentiveAccidentLSTM

# ── Load models (cached) ─────────────────────────────────
_yolo_model = None
_lstm_model = None

@st.cache_resource
def load_models():
    global _yolo_model, _lstm_model
    if _yolo_model is None or _lstm_model is None:
        yolo = None
        lstm = None
        if Path(YOLO_WEIGHTS).exists():
            yolo = YOLO(YOLO_WEIGHTS)
        if Path(LSTM_WEIGHTS).exists():
            lstm = AccidentLSTM(input_dim=FEATURE_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(get_device())
            lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=get_device()))
            lstm.eval()
        _yolo_model = yolo
        _lstm_model = lstm
    return _yolo_model, _lstm_model

# ── Vehicle detector for bounding boxes ─────────────────
_vehicle_detector = None

@st.cache_resource
def load_vehicle_detector():
    global _vehicle_detector
    if _vehicle_detector is None:
        _vehicle_detector = YOLO("models/helmet_detection/weights/yolov8n.pt")
    return _vehicle_detector
