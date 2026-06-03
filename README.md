# 🛡️ Aegis Eye — Real-time Autonomous Accident Detection & Localization System

Aegis Eye is a production-grade Computer Vision and Deep Learning capstone project designed to autonomously monitor traffic camera streams, detect vehicle collisions in real time, freeze operational feeds at the timestamp of the threat, isolate the accident spot via thick red warning highlight boxes, and geolocalize the collision on an interactive map for emergency services dispatch.

---

## 🏗️ Repository Architecture

The repository has been restructured into a clean, modular, and professional folder hierarchy, separating model training, inference pipelines, utilities, testing, and deployment:

```
accident_detection/
├── data/                  # Datasets (raw & processed UCF-Crime clips)
├── models/                # Saved LSTM and YOLO model weights (.pt files)
├── logs/                  # SQLite incident registry database and snapshot images
├── runs/                  # YOLOv8 classification training runs
├── pipeline/              # Preprocessing & core inference engines
│   ├── preprocess.py      # Video frame extractor and manifest builder
│   └── detector.py        # Pipeline threat detection engine
├── training/              # Deep Learning model training scripts
│   ├── train_yolo.py      # YOLOv8 classifier fine-tuning
│   └── train_lstm.py      # LSTM ResNet18 sequence model training
├── tools/                 # Data utility and cleansing scripts
│   ├── fix_data.py        # Directory structure alignment script
│   └── fix_data_leakage.py # Dataset overlap correction tool
├── tests/                 # Threshold and pipeline verification
│   └── test_thresholds.py # Real-time score visualizer and tuner
├── alerts/                # Integration notification modules
├── app.py                 # Main Streamlit unified command center entry point
├── setup.py               # Repository installer and workspace initializer
└── requirements.txt       # Python dependency manifest
```

---

## 🛰️ System Threat Pipeline Flow

The system processes CCTV feeds frame-by-frame and coordinates threat response automatically:

```
                  ┌───────────────────────┐
                  │   CCTV / Camera Feed  │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │ Preprocess & Skip     │
                  └───────────┬───────────┘
                              ▼
                 ┌────────────┴────────────┐
                 │ YOLOv8 Classify         │
                 │   - Scene classification│
                 │ LSTM Sequence check     │
                 │   - Temporal anomalies  │
                 └────────────┬────────────┘
                              ▼
                 ┌────────────┴────────────┐
                 │    Collision Detected?  │
                 └──────┬────────────┬─────┘
                 Normal │            │ Yes (Both threshold crossed)
                        ▼            ▼
         ┌──────────────────┐    ┌──────────────────────────────────┐
         │ Track vehicles in│    │ 1. Freeze Video Feed             │
         │ elegant Cyan boxes│    │ 2. Detect & Highlight spot box   │
         │ Keep stream live │    │ 3. Log to SQLite with timestamps  │
         └──────────────────┘    │ 4. Procedurally generate GPS      │
                                 │ 5. Plot Red Pin on st.map        │
                                 └──────────────────────────────────┘
```

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Ensure Python 3.10+ and a CUDA-compatible environment (if using GPU acceleration) are installed.

### 2. Environment Initialization
Clone the repository, initialize a Python virtual environment, and install dependencies:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Workspace Setup
Initialize directories, log databases, and verify weights:
```powershell
python setup.py
```

---

## 🧪 Operational Workflow

### Phase 1: Data Preprocessing
Extract frames and create the dataset manifest for training:
```powershell
python pipeline/preprocess.py
```

### Phase 2: YOLOv8 Classifier Fine-Tuning
Train the YOLOv8-cls network to identify accident vs normal frames:
```powershell
python training/train_yolo.py
```

### Phase 3: Temporal LSTM Training
Train the ResNet18-backboned LSTM to model sequential temporal anomalies:
```powershell
python training/train_lstm.py
```

### Phase 4: Threshold Testing & Diagnostics
Calibrate pipeline trigger levels by feeding a pre-recorded clip and checking real-time scores:
```powershell
python tests/test_thresholds.py --source "data/raw/accident/RoadAccidents080_x264.mp4"
```

---

## 🖥️ Command Center Deployment

Launch the high-tech, dark-themed operational command center:
```powershell
streamlit run app.py
```

### Key Dashboard Features:
1. **Multi-Source Select**: Connect a live USB webcam, an RTSP network stream address, or select from a dropdown listing high-res demo clips (`RoadAccidents080_x264.mp4` etc.) directly.
2. **Automatic Feed Freeze**: Pauses and locks the stream on the exact millisecond a collision is registered, capturing the telemetry profile.
3. **Collision Spot Localization**: Uses the vehicle detector to isolate the exact car/spot involved and draws an extra-thick, glowing red box with a `💥 ACCIDENT SPOT TRIGGER` overlay on the frozen frame.
4. **GPS Beacon Localization**: Renders an interactive map indicating the exact location of the crash, alongside a dispatch coordinates telemetry display.
5. **Database Registry History Lookup**: Browse logged historical threats via an inspection dropdown. Selecting any entry immediately loads its archived coordinates, confidence scores, and raw frozen snapshot for retrospective analysis.
6. **Alert Reset Override**: Clear active alarms and hot-swap cameras or resume tracking with a single click.
