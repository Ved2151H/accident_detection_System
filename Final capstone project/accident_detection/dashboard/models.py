import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from dashboard.config import (
    YOLO_WEIGHTS, LSTM_WEIGHTS, DEVICE, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES
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

# ── Load models (cached) ─────────────────────────────────
@st.cache_resource
def load_models():
    yolo = None
    lstm = None
    if Path(YOLO_WEIGHTS).exists():
        yolo = YOLO(YOLO_WEIGHTS)
    if Path(LSTM_WEIGHTS).exists():
        lstm = AccidentLSTM().to(DEVICE)
        lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=DEVICE))
        lstm.eval()
    return yolo, lstm

# ── Vehicle detector for bounding boxes ─────────────────
@st.cache_resource
def load_vehicle_detector():
    return YOLO("yolov8n.pt")
