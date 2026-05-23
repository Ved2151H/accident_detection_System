"""
Day 5 — Sleek Single-Screen Command Center for Live Accident Detection
Features:
  - Unified premium dark mode dashboard layout
  - Real-time video/webcam feed analysis
  - Live vehicle tracking telemetry (Cyan bounding boxes)
  - Automatic feed freezing on accident detection
  - Accident spot localization (Thick glowing red collision squares)
  - Procedural/Realistic GPS coordinates logging
  - Live interactive map integration (st.map)
  - Incident database history lookup and telemetry inspection
Run: streamlit run day5_dashboard.py
"""

import sqlite3
import os
import cv2
import time
import datetime
import collections
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

# Aegis Eye Premium DIGIPIN & Calibration imports
from tools.digipin_helper import DigiPinHelper

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
FRAME_SKIP     = 3
ACCIDENT_CONF  = 0.15
LSTM_THRESHOLD = 0.70
DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────

# Initialize DIGIPIN Helper Instance
digipin_helper = DigiPinHelper()

# ── Adaptive Statistical Calibration Engine ──────────────
class AdaptiveThreatCalibrator:
    def __init__(self, window_size=30, base_k_yolo=2.4, base_k_lstm=1.9):
        self.window_size = window_size
        self.k_yolo = base_k_yolo
        self.k_lstm = base_k_lstm
        self.yolo_history = []
        self.lstm_history = []
        self.default_yolo = ACCIDENT_CONF
        self.default_lstm = LSTM_THRESHOLD
        
    def set_rigor(self, rigor_label):
        if rigor_label == "Aggressive (Low Latency)":
            self.k_yolo = 1.6
            self.k_lstm = 1.4
        elif rigor_label == "Conservative (Low False-Positives)":
            self.k_yolo = 3.2
            self.k_lstm = 2.4
        else: # Standard (Recommended)
            self.k_yolo = 2.4
            self.k_lstm = 1.9

    def update(self, yolo_score, lstm_score):
        # Exclude active collision segments to avoid baseline contamination
        if yolo_score < 0.40:
            self.yolo_history.append(yolo_score)
            if len(self.yolo_history) > self.window_size:
                self.yolo_history.pop(0)
        if lstm_score < 0.85:
            self.lstm_history.append(lstm_score)
            if len(self.lstm_history) > self.window_size:
                self.lstm_history.pop(0)

    def get_thresholds(self):
        if len(self.yolo_history) >= 8:
            mean_yolo = np.mean(self.yolo_history)
            std_yolo = np.std(self.yolo_history)
            dyn_yolo = max(0.12, mean_yolo + self.k_yolo * std_yolo)
            dyn_yolo = min(0.35, dyn_yolo)
        else:
            dyn_yolo = self.default_yolo
            
        if len(self.lstm_history) >= 8:
            mean_lstm = np.mean(self.lstm_history)
            std_lstm = np.std(self.lstm_history)
            dyn_lstm = max(0.60, mean_lstm + self.k_lstm * std_lstm)
            dyn_lstm = min(0.85, dyn_lstm)
        else:
            dyn_lstm = self.default_lstm
            
        return round(dyn_yolo, 3), round(dyn_lstm, 3)

# ── Custom Premium Dark Leaflet Cartography HTML ─────────
def get_leaflet_html(lat, lon, digipin, is_incident=False, zoom=14):
    color_theme = "#ff3333" if is_incident else "#00ffff"
    popup_title = "🚨 ACTIVE ACCIDENT ANOMALY" if is_incident else "🛰️ BASE STATION MONITOR"
    badge_style = (
        "background: linear-gradient(135deg, #ff8c00, #ff4500); border: 1px solid #ffaa66;"
        if is_incident else
        "background: linear-gradient(135deg, #008080, #004d4d); border: 1px solid #00aaaa;"
    )
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Aegis Eye Live Geolocation</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body, #map {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                background-color: #06060c;
            }}
            /* Premium Cyberpunk Leaflet Customizations */
            .leaflet-popup-content-wrapper {{
                background: #0b0b14 !important;
                border: 1px solid {color_theme} !important;
                color: #e0e0ea !important;
                border-radius: 12px !important;
                font-family: monospace !important;
                padding: 8px !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6) !important;
            }}
            .leaflet-popup-tip {{
                background: #0b0b14 !important;
                border: 1px solid {color_theme} !important;
            }}
            .popup-title {{
                color: {color_theme};
                font-weight: 800;
                font-size: 1.05rem;
                letter-spacing: 1px;
                border-bottom: 1px solid rgba(255,255,255,0.08);
                padding-bottom: 6px;
                margin-bottom: 10px;
                text-transform: uppercase;
            }}
            .popup-item {{
                margin: 6px 0;
                font-size: 0.9rem;
                line-height: 1.4;
            }}
            .digipin-badge-popup {{
                display: block;
                {badge_style}
                color: #ffffff;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 1rem;
                font-weight: 800;
                text-align: center;
                letter-spacing: 1px;
                margin-top: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            /* Custom Pulsing Target Marker styling */
            .pulsing-marker {{
                position: relative;
            }}
            .pin {{
                width: 12px;
                height: 12px;
                background-color: {color_theme};
                border-radius: 50%;
                border: 2px solid #fff;
                box-shadow: 0 0 10px {color_theme};
                position: absolute;
                top: 6px;
                left: 6px;
            }}
            .pulse {{
                width: 24px;
                height: 24px;
                border: 2px solid {color_theme};
                border-radius: 50%;
                position: absolute;
                top: 0px;
                left: 0px;
                animation: pulsate 1.5s ease-out infinite;
                opacity: 0;
            }}
            @keyframes pulsate {{
                0% {{ transform: scale(0.1); opacity: 0.0; }}
                50% {{ opacity: 0.8; }}
                100% {{ transform: scale(1.3); opacity: 0.0; }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                zoomControl: true,
                attributionControl: false
            }}).setView([{lat}, {lon}], {zoom});

            // CartoDB DarkMatter - custom cyber aesthetic tiles
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                maxZoom: 20
            }}).addTo(map);

            // Pulsing target DivIcon
            var pulsingIcon = L.divIcon({{
                className: 'pulsing-marker',
                html: '<div class="pulse"></div><div class="pin"></div>',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }});

            var marker = L.marker([{lat}, {lon}], {{ icon: pulsingIcon }}).addTo(map);

            var popupContent = f`
                <div class="popup-title">{popup_title}</div>
                <div class="popup-item"><b>LATITUDE:</b> {lat:.6f}° N</div>
                <div class="popup-item"><b>LONGITUDE:</b> {lon:.6f}° E</div>
                <div class="popup-item">
                    <b>INDIA POST DIGIPIN:</b>
                    <span class="digipin-badge-popup">🇮🇳 {digipin}</span>
                </div>
                {'<div class="popup-item" style="color: #ff3333; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">📡 DISPATCHING GPS TRACKER BEACON</div>' if is_incident else '<div class="popup-item" style="color: #00ffff; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">🟢 PATROL BEACONS SECURE</div>'}
            `;

            marker.bindPopup(popupContent).openPopup();
        </script>
    </body>
    </html>
    """
    return html

st.set_page_config(
    page_title = "Aegis Eye — Accident Detection & Localization System",
    page_icon  = "🚨",
    layout     = "wide",
)

# Custom High-End Cyberpunk Theme & CSS Styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Share+Tech+Mono&display=swap');

/* Global resets and background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #06060c;
    color: #e0e0ea;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stHeader"] {
    background-color: rgba(6, 6, 12, 0.8);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] {
    background-color: #0b0b14;
    border-right: 1px solid #1f1f2e;
}

/* Sidebar styling */
.sidebar-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    color: #ffffff;
    font-size: 1.6rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.sidebar-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #00ffff;
    letter-spacing: 2px;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Header & Telemetry Panels */
.telemetry-card {
    background: rgba(18, 18, 30, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}

.card-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    letter-spacing: 1.5px;
    color: #00ffff;
    margin-bottom: 12px;
    font-weight: 600;
    text-transform: uppercase;
    border-left: 3px solid #00ffff;
    padding-left: 8px;
}

/* Status Banners */
.accident-banner-red {
    background: linear-gradient(135deg, #7c0000 0%, #3a0000 100%);
    border: 2px solid #ff3333;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(255, 51, 51, 0.4);
    animation: glow-red 1.5s infinite alternate;
}

@keyframes glow-red {
    from { box-shadow: 0 0 10px rgba(255, 51, 51, 0.2); }
    to { box-shadow: 0 0 25px rgba(255, 51, 51, 0.6); }
}

.normal-banner-green {
    background: linear-gradient(135deg, #004d26 0%, #002411 100%);
    border: 2px solid #00e676;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);
}

/* Metric text */
.metric-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    margin-top: 4px;
}

.coord-display {
    background: #0d0d18;
    border-left: 4px solid #ff3333;
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #ff3333;
    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
}

/* Global dividers */
.divider-neon {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

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

# ── DB helper functions with schema migration ─────────────
def init_db():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, source TEXT,
        yolo_conf REAL, lstm_prob REAL, snapshot TEXT,
        latitude REAL, longitude REAL, digipin TEXT)""")
    
    # Database migration checks
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(incidents)")
        columns = [row[1] for row in cursor.fetchall()]
        if "latitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN latitude REAL")
        if "longitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN longitude REAL")
        if "digipin" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN digipin TEXT")
    except Exception as e:
        print(f"DB Migration Error: {e}")
    conn.commit()
    conn.close()

def log_incident(source, yolo_conf, lstm_prob, snapshot_path, latitude, longitude, digipin):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO incidents (timestamp, source, yolo_conf, lstm_prob, snapshot, latitude, longitude, digipin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), source, yolo_conf, lstm_prob, snapshot_path, latitude, longitude, digipin))
    conn.commit()
    conn.close()

def get_incidents():
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", conn)
    conn.close()
    return df

def save_snapshot(frame):
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOTS_DIR, f"incident_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

# ── Realistic GPS Coordinate Generator ───────────────────
def generate_gps():
    # Centered in Pune, India
    base_lat = 18.5204
    base_lon = 73.8567
    offset_lat = np.random.uniform(-0.015, 0.015)
    offset_lon = np.random.uniform(-0.015, 0.015)
    return round(base_lat + offset_lat, 6), round(base_lon + offset_lon, 6)

# ── Core Threat Scan Inference on Single Frame ───────────
def run_inference(frame, buffer, yolo, lstm, accident_conf_val=ACCIDENT_CONF, lstm_threshold_val=LSTM_THRESHOLD):
    """Detects accidents and tracks vehicles using YOLO + LSTM."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = frame_transform(rgb)
    buffer.append(tensor)

    # YOLO classification (accident vs normal)
    results  = yolo(frame, verbose=False)
    top_cls  = results[0].probs.top1
    top_conf = results[0].probs.top1conf.item()
    yolo_acc = top_conf if top_cls == 0 else (1 - top_conf)

    # LSTM Temporal sequence validation
    lstm_acc = 0.0
    if len(buffer) == SEQUENCE_LEN:
        seq = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits   = lstm(seq)
            probs    = torch.softmax(logits, dim=1)[0]
        lstm_acc = probs[0].item()

    accident = (yolo_acc >= accident_conf_val and lstm_acc >= lstm_threshold_val)

    # Vehicle Tracking Overlay (Cyan tech bounding boxes)
    detector    = load_vehicle_detector()
    cls_ids     = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)
    det_results = detector(frame, verbose=False, classes=cls_ids)
    boxes       = det_results[0].boxes
    cls_names   = {2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}

    if boxes is not None and len(boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf            = box.conf[0].item()
            cls_id          = int(box.cls[0].item())
            name            = cls_names.get(cls_id, "Vehicle")

            # Draw elegant telemetry-cyan box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return accident, yolo_acc, lstm_acc, frame

# ── Session State Initialization ────────────────────────
if "stream_running" not in st.session_state:
    st.session_state.stream_running = False
if "accident_frozen" not in st.session_state:
    st.session_state.accident_frozen = False
if "current_coords" not in st.session_state:
    st.session_state.current_coords = None
if "frozen_frame" not in st.session_state:
    st.session_state.frozen_frame = None
if "current_incident_data" not in st.session_state:
    st.session_state.current_incident_data = None
if "calibrator" not in st.session_state:
    st.session_state.calibrator = AdaptiveThreatCalibrator()

# Initialize Database on boot
init_db()
yolo_model, lstm_model = load_models()

if not yolo_model or not lstm_model:
    st.error("Model weights missing. Ensure YOLO and LSTM weights are properly loaded.")
    st.code(f"Missing weight paths:\n - {YOLO_WEIGHTS}\n - {LSTM_WEIGHTS}")
    st.stop()

# ════════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">🚨 AEGIS EYE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">ACCIDENT LOCALIZATION & DISPATCH</div>', unsafe_allow_html=True)
    st.divider()

    # Source Selection
    source_type = st.selectbox("Select Camera Input Stream", ["📷 Laptop / USB Webcam", "📁 High-Res Demo footage", "📡 CCTV RTSP Stream"])
    video_source = None
    source_name = ""

    if source_type == "📷 Laptop / USB Webcam":
        cam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
        video_source = cam_index
        source_name = f"Webcam {cam_index}"
    elif source_type == "📁 High-Res Demo footage":
        # Look for existing video files inside data/raw/accident
        accident_dir = Path("data/raw/accident")
        video_files = []
        if accident_dir.exists():
            video_files = list(accident_dir.glob("*.mp4")) + list(accident_dir.glob("*.avi"))
        
        if video_files:
            selected_file = st.selectbox("Select Demo Video Clip", [f.name for f in video_files])
            video_source = str(accident_dir / selected_file)
            source_name = f"Demo clip: {selected_file}"
        else:
            # Fallback upload
            uploaded = st.file_uploader("Upload footage file", type=["mp4", "avi"])
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
                tfile.write(uploaded.read())
                tfile.close()
                video_source = tfile.name
                source_name = f"Upload: {uploaded.name}"
    else:
        video_source = st.text_input("Enter RTSP stream address", "rtsp://192.168.1.100:554/stream")
        source_name = "RTSP Network Stream"

    st.divider()
    st.markdown("**⚙️ AUTOMATIC THREAT CALIBRATION**")
    auto_calib = st.toggle("Enable AI Self-Calibration", value=True, help="Automatically adjust detection thresholds based on real-time background noise.")
    
    if auto_calib:
        rigor = st.selectbox(
            "Detection Rigor / Sensitivity",
            ["Standard (Recommended)", "Aggressive (Low Latency)", "Conservative (Low False-Positives)"],
            help="Adjust statistical deviation multipliers for dynamic thresholding."
        )
        st.session_state.calibrator.set_rigor(rigor)
        # Fetch dynamically calculated thresholds
        accident_conf, lstm_threshold = st.session_state.calibrator.get_thresholds()
        
        st.info(f"🤖 Dynamic YOLO Thr: {accident_conf:.1%}\n\n🤖 Dynamic LSTM Thr: {lstm_threshold:.1%}")
    else:
        accident_conf = st.slider("YOLO Accident Threshold", min_value=0.05, max_value=0.95, value=0.15, step=0.01, help="Calibrate YOLO classification trigger level.")
        lstm_threshold = st.slider("LSTM Anomaly Threshold", min_value=0.05, max_value=0.95, value=0.70, step=0.01, help="Calibrate sequential temporal trigger level.")

    st.divider()

    # System Buttons
    col_start, col_stop = st.columns(2)
    start_sys = col_start.button("▶ START SYSTEM", type="primary", use_container_width=True)
    stop_sys = col_stop.button("⏹ STOP SYSTEM", use_container_width=True)

    if start_sys:
        st.session_state.stream_running = True
        st.session_state.accident_frozen = False
        st.session_state.frozen_frame = None
        st.session_state.current_coords = None
        st.session_state.current_incident_data = None
        st.rerun()

    if stop_sys:
        st.session_state.stream_running = False
        st.session_state.accident_frozen = False
        st.session_state.frozen_frame = None
        st.session_state.current_coords = None
        st.session_state.current_incident_data = None
        st.rerun()

    # Model Telemetry Information
    st.divider()
    st.markdown("**🛡️ SYSTEM HARDWARE METRICS**")
    st.success(f"YOLOv8 Engine: Loaded ✓")
    st.success(f"LSTM Core: Loaded ✓")
    st.info(f"Target Processor: {str(DEVICE).upper()}")

# ════════════════════════════════════════════════════════
# MAIN SCREEN LAYOUT
# ════════════════════════════════════════════════════════
left_pane, right_pane = st.columns([0.65, 0.35])

# Left Pane: Main Video Stream & Telemetry Banners
with left_pane:
    st.markdown("<div class='card-header'>🛡️ LIVE MONITORING AND THREAT DETECTOR</div>", unsafe_allow_html=True)
    video_container = st.empty()
    status_container = st.empty()

# Right Pane: Map coordinates & Historical Database logs
with right_pane:
    st.markdown("<div class='card-header'>📍 LIVE GEOLOCATION & DISPATCH</div>", unsafe_allow_html=True)
    coords_container = st.empty()
    map_container = st.empty()
    
    st.divider()
    st.markdown("<div class='card-header'>📋 INCIDENT DATABASE REGISTER</div>", unsafe_allow_html=True)
    db_container = st.empty()

# ════════════════════════════════════════════════════════
# CORE SYSTEM LOOP AND CONTROLLER
# ════════════════════════════════════════════════════════
def update_telemetry_standby(container, yolo_acc, lstm_acc, yolo_thr=0.20, lstm_thr=0.48):
    with container:
        st.markdown(f"""
        <div class="normal-banner-green">
            <h4 style="margin: 0; color: #00e676;">🟢 ACTIVE THREAT SCANNER</h4>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.9;">
                Inference models analyzing feed in real-time. No threat anomalies detected.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class="telemetry-card" style="border-color: rgba(0, 230, 118, 0.2); position: relative;">
            <span style="color: #00e676; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">YOLO NOISE LEVEL</span>
            <div class="metric-val" style="color: #00e676;">{yolo_acc:.2%}</div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 4px;">Trigger Threshold: {yolo_thr:.0%}</div>
            <div style="background-color: #11111a; border-radius: 4px; height: 8px; margin-top: 6px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); position: relative;">
                <div style="background: linear-gradient(95deg, #00e676, #00ffff); width: {yolo_acc * 100}%; height: 100%;"></div>
                <div style="position: absolute; left: {yolo_thr * 100}%; top: 0; width: 2px; height: 100%; background-color: #ff3333; box-shadow: 0 0 4px #ff3333;" title="Threshold"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class="telemetry-card" style="border-color: rgba(0, 230, 118, 0.2); position: relative;">
            <span style="color: #00e676; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">LSTM ANOMALY RATE</span>
            <div class="metric-val" style="color: #00e676;">{lstm_acc:.2%}</div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 4px;">Trigger Threshold: {lstm_thr:.0%}</div>
            <div style="background-color: #11111a; border-radius: 4px; height: 8px; margin-top: 6px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); position: relative;">
                <div style="background: linear-gradient(95deg, #00e676, #00ffff); width: {lstm_acc * 100}%; height: 100%;"></div>
                <div style="position: absolute; left: {lstm_thr * 100}%; top: 0; width: 2px; height: 100%; background-color: #ff3333; box-shadow: 0 0 4px #ff3333;" title="Threshold"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── DB Feed and Inspection Render ──
df_db = get_incidents()
with db_container:
    if df_db.empty:
        st.info("No threat logs found in database. Start monitoring to log events.")
    else:
        options = ["-- Active Monitor --"] + [f"Incident #{row['id']} ({row['timestamp'][11:19]})" for _, row in df_db.iterrows()]
        selected_db_log = st.selectbox("🔍 Select Log to Inspect Spot Map/Snapshot:", options)
        
        if selected_db_log != "-- Active Monitor --":
            # Extract incident details
            inc_id = int(selected_db_log.split("#")[1].split(" ")[0])
            row = df_db[df_db["id"] == inc_id].iloc[0]
            
            st.session_state.accident_frozen = True
            st.session_state.stream_running = False
            
            snap_path = row["snapshot"]
            if snap_path and Path(snap_path).exists():
                try:
                    img = Image.open(snap_path)
                    st.session_state.frozen_frame = img
                except Exception:
                    st.session_state.frozen_frame = None
            else:
                st.session_state.frozen_frame = None
                
            # If historical incident lacks DIGIPIN, dynamically encode it on loading
            digipin_val = row.get("digipin") if "digipin" in row else None
            if not digipin_val or pd.isna(digipin_val):
                digipin_val = digipin_helper.gps_to_digipin(row["latitude"], row["longitude"])
                
            st.session_state.current_coords = (row["latitude"], row["longitude"], digipin_val)
            st.session_state.current_incident_data = {
                "yolo_conf": row["yolo_conf"],
                "lstm_prob": row["lstm_prob"],
                "source": row["source"],
                "timestamp": row["timestamp"]
            }

# ── Render Screen depending on System state ──
if st.session_state.accident_frozen:
    # Render frozen accident state
    if st.session_state.frozen_frame is not None:
        video_container.image(st.session_state.frozen_frame, use_column_width=True)
    else:
        video_container.warning("Incident snapshot frame not found on disk.")

    with status_container:
        data = st.session_state.current_incident_data
        st.markdown(f"""
        <div class="accident-banner-red">
            <h3 style="margin: 0; color: white;">🚨 COLLISION DETECTED & SYSTEM FROZEN</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                An active vehicle accident threat was detected on <b>{data['source']}</b> at <b>{data['timestamp']}</b>.
                The camera feed has been frozen at the anomaly timestamp. Automatic rescue dispatch localizing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.markdown(f"""
        <div class="telemetry-card" style="border-color: rgba(255, 51, 51, 0.3);">
            <span style="color: #ff3333; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">YOLO COLLISION PROBABILITY</span>
            <div class="metric-val" style="color: #ff3333;">{data['yolo_conf']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_m2.markdown(f"""
        <div class="telemetry-card" style="border-color: rgba(255, 51, 51, 0.3);">
            <span style="color: #ff3333; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">LSTM TEMPORAL THREAT PROBABILITY</span>
            <div class="metric-val" style="color: #ff3333;">{data['lstm_prob']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔓 CLEAR ACTIVE ALERT & RESUME MONITORING", type="primary", use_container_width=True):
            st.session_state.accident_frozen = False
            st.session_state.frozen_frame = None
            st.session_state.current_coords = None
            st.session_state.current_incident_data = None
            st.session_state.stream_running = True
            st.rerun()

elif st.session_state.stream_running:
    # Active frame loop processing
    if video_source is None:
        video_container.error("Please configure video source details first.")
        st.session_state.stream_running = False
    else:
        src_val = int(video_source) if str(video_source).isdigit() else video_source
        cap = cv2.VideoCapture(src_val)
        
        if not cap.isOpened():
            video_container.error(f"Cannot initialize camera stream: {video_source}")
            st.session_state.stream_running = False
        else:
            buffer = collections.deque(maxlen=SEQUENCE_LEN)
            frame_idx = 0
            
            while cap.isOpened() and st.session_state.stream_running:
                ret, frame = cap.read()
                if not ret:
                    video_container.info("Stream source reached end of footage.")
                    st.session_state.stream_running = False
                    break
                
                frame_idx += 1
                if frame_idx % FRAME_SKIP != 0:
                    continue
                
                # Inference
                accident, yolo_acc, lstm_acc, annotated = run_inference(
                    frame.copy(), buffer, yolo_model, lstm_model, accident_conf, lstm_threshold)
                
                # Update automated statistical baseline
                if auto_calib:
                    st.session_state.calibrator.update(yolo_acc, lstm_acc)
                
                # Show live video frame
                rgb_live = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_container.image(rgb_live, use_column_width=True)
                
                # Threat detected event trigger
                if accident:
                    lat, lon = generate_gps()
                    digipin_code = digipin_helper.gps_to_digipin(lat, lon)
                    
                    # Highlight vehicle spot in RED thick square box on the frozen frame
                    detector    = load_vehicle_detector()
                    cls_ids     = [2, 3, 5, 7]
                    det_results = detector(frame, verbose=False, classes=cls_ids)
                    boxes       = det_results[0].boxes
                    
                    highlight_frame = frame.copy()
                    h, w = highlight_frame.shape[:2]
                    
                    if boxes is not None and len(boxes):
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Clip to frame boundaries to avoid out-of-bounds assertions
                            x1 = max(0, min(x1, w - 1))
                            x2 = max(0, min(x2, w - 1))
                            y1 = max(0, min(y1, h - 1))
                            y2 = max(0, min(y2, h - 1))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            # Draw thick bright red square highlighting crash zone
                            cv2.rectangle(highlight_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            
                            # Solid red corners
                            clen = 22
                            for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                                dx = clen  if cx == x1 else -clen
                                dy = clen  if cy == y1 else -clen
                                cv2.line(highlight_frame,(cx,cy),(cx+dx,cy),(0,0,255),5)
                                cv2.line(highlight_frame,(cx,cy),(cx,cy+dy),(0,0,255),5)
                            
                            # Glowing red background overlay for the box
                            sub_r = highlight_frame[y1:y2, x1:x2]
                            if sub_r.size > 0:
                                glow_r = np.zeros(sub_r.shape, dtype=np.uint8)
                                glow_r[:, :] = [0, 0, 180]
                                highlight_frame[y1:y2, x1:x2] = cv2.addWeighted(sub_r, 0.7, glow_r, 0.3, 0)
                            
                            # Label crash box
                            label_str = "💥 ACCIDENT SPOT TRIGGER"
                            (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(highlight_frame, (x1, max(0, y1-lh-8)), (x1+lw+6, y1), (0, 0, 255), -1)
                            cv2.putText(highlight_frame, label_str, (x1+3, max(lh, y1-3)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    
                    # Highlight whole screen red border & tint
                    cv2.rectangle(highlight_frame, (0, 0), (w, h), (0, 0, 255), 8)
                    tint = highlight_frame.copy()
                    cv2.rectangle(tint, (0,0), (w,h), (0,0,150), -1)
                    cv2.addWeighted(tint, 0.1, highlight_frame, 0.9, 0, highlight_frame)
                    
                    # Save incident files
                    snap_path = save_snapshot(highlight_frame)
                    log_incident(source_name, yolo_acc, lstm_acc, snap_path, lat, lon, digipin_code)
                    
                    # Store threat session details
                    st.session_state.accident_frozen = True
                    st.session_state.frozen_frame = cv2.cvtColor(highlight_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.current_coords = (lat, lon, digipin_code)
                    st.session_state.current_incident_data = {
                        "yolo_conf": yolo_acc,
                        "lstm_prob": lstm_acc,
                        "source": source_name,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    cap.release()
                    st.rerun()
                
                # Update Real-time scanning analytics
                update_telemetry_standby(status_container, yolo_acc, lstm_acc, accident_conf, lstm_threshold)
                time.sleep(0.01)
                
            cap.release()
else:
    # Standby monitoring system idle state
    video_container.info("⚙️ Threat monitoring engine standby. Select a feed input and press '▶ START SYSTEM' in the control panel to begin.")
    
    with status_container:
        st.markdown("""
        <div class="normal-banner-green" style="background: rgba(10, 25, 15, 0.3); border-color: #333344;">
            <h4 style="margin: 0; color: #888;">🛰️ STANDBY / AWAITING SIGNAL</h4>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #777788;">
                Inference core initialized. Ready to begin threat scanning.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# GEOLOCATION DISPATCH MAP RENDERING
# ════════════════════════════════════════════════════════
if st.session_state.current_coords:
    lat, lon = st.session_state.current_coords
    with coords_container:
        st.markdown(f"""
        <div class="coord-display">
            💥 LIVE CRASH VEHICLE LOCALIZATION:<br>
            <b>LATITUDE:</b> {lat:.6f}° N<br>
            <b>LONGITUDE:</b> {lon:.6f}° E<br>
            <span style="font-size: 0.8rem; color: #ff3333; opacity: 0.8; font-weight: bold; animation: pulse 1s infinite;">
                📡 DISPATCHING GPS TRACKER BEACON TO SPOT
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Render map
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    with map_container:
        st.map(map_df, zoom=14)
else:
    # Standby central monitoring map view
    with coords_container:
        st.markdown("""
        <div class="coord-display" style="border-left-color: #00ffff; color: #00ffff; background: #0f1620;">
            🛰️ CENTRAL MONITORING BASE STATION:<br>
            <b>LATITUDE:</b> 18.520400° N<br>
            <b>LONGITUDE:</b> 73.856700° E<br>
            <span style="font-size: 0.8rem; color: #00ffff; opacity: 0.8;">
                🟢 ALL PATROL BEACONS SECURE
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    map_df = pd.DataFrame({"lat": [18.5204], "lon": [73.8567]})
    with map_container:
        st.map(map_df, zoom=12)