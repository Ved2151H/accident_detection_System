import numpy as np

# ── Weight Paths & Model Configuration ───────────────────
YOLO_WEIGHTS   = "models/accident_detection/checkpoints/classify/models/accident_detector/weights/best.pt"
LSTM_WEIGHTS   = "models/accident_detection/weights/lstm_best.pt"
DB_PATH        = "logs/incidents.db"
SNAPSHOTS_DIR  = "logs/snapshots"
SEQUENCE_LEN   = 8
FEATURE_DIM    = 512
HIDDEN_SIZE    = 64
NUM_LAYERS     = 1
NUM_CLASSES    = 2
FRAME_SKIP     = 3
ACCIDENT_CONF  = 0.20
LSTM_THRESHOLD = 0.65

# Lazy device loader
_device = None
def get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _device

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
