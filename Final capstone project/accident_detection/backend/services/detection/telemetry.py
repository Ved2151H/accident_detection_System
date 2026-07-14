import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import os
import urllib.request
import json
from pathlib import Path
from collections import deque
from backend.utils.config import ACCIDENT_CONF, LSTM_THRESHOLD, SEQUENCE_LEN, get_device, DB_PATH
from backend.services.detection.models import frame_transform, load_vehicle_detector
from ultralytics import YOLO

# ── MLP Fusion Classifier Definition ───────────────────
class AccidentFusionMLP(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ── Automatic Fire/Smoke Model Downloader ──────────────
def download_fire_smoke_model(target_path="models/fire_smoke_model.pt"):
    if os.path.exists(target_path):
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        print("Attempting download via huggingface_hub...")
        downloaded_path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
        import shutil
        shutil.copy(downloaded_path, target_path)
        print(f"Model saved successfully to {target_path} via huggingface_hub")
        return target_path
    except Exception as hf_err:
        print(f"huggingface_hub download failed or not installed: {hf_err}. Trying direct download...")
        
    url = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
    print(f"Downloading fire/smoke model from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Model saved successfully to {target_path} via direct download")
        return target_path
    except Exception as e:
        print(f"Error downloading fire/smoke model: {e}")
        return None

_fire_smoke_detector = None

@st.cache_resource
def load_fire_smoke_detector():
    global _fire_smoke_detector
    if _fire_smoke_detector is not None:
        return _fire_smoke_detector
    path = download_fire_smoke_model()
    if path:
        try:
            _fire_smoke_detector = YOLO(path)
            return _fire_smoke_detector
        except Exception:
            return None
    return None

_mlp_model = None
_mlp_scaler = None

@st.cache_resource
def load_mlp_model():
    global _mlp_model, _mlp_scaler
    if _mlp_model is not None:
        return _mlp_model, _mlp_scaler
    mlp_path = "models/accident_detection/weights/fusion_mlp.pt"
    scaler_path = "models/accident_detection/weights/fusion_scaler.json"
    if os.path.exists(mlp_path) and os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'r') as f:
                scaler = json.load(f)
            model = AccidentFusionMLP(hidden_dim=32)
            model.load_state_dict(torch.load(mlp_path, map_location=get_device()))
            model.to(get_device())
            model.eval()
            _mlp_model = model
            _mlp_scaler = scaler
            return model, scaler
        except Exception as e:
            print(f"Error loading MLP model: {e}")
            return None, None
    return None, None

# ── Road Lane Detection ─────────────────────────────────
def detect_lanes(frame):
    h, w = frame.shape[:2]
    # Resize frame to a standard width of 320 pixels for ultra-fast processing
    scale = 320.0 / w
    small_frame = cv2.resize(frame, (320, int(h * scale)))
    sh, sw = small_frame.shape[:2]
    roi = small_frame[int(sh * 0.5):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Adjust minLineLength and maxLineGap to scale
    min_line_len = max(5, int(30 * scale))
    max_line_gap = max(3, int(20 * scale))
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=min_line_len, maxLineGap=max_line_gap)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            angles.append(angle)
            
    return np.mean(angles) if angles else 90.0

# ── Camera Motion Estimation & Compensation ─────────────
def estimate_camera_motion(prev_gray, curr_gray):
    if prev_gray is None or curr_gray is None or prev_gray.shape != curr_gray.shape:
        return 0.0, 0.0
    prev_f = np.float32(prev_gray)
    curr_f = np.float32(curr_gray)
    (dx, dy), _ = cv2.phaseCorrelate(prev_f, curr_f)
    return dx, dy

def compute_compensated_flow(prev_gray, curr_gray, dx, dy):
    if prev_gray is None or curr_gray is None:
        return 0.0, 0.0, 0.0, None
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[..., 0] -= dx
    flow[..., 1] -= dy
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_magnitude = np.mean(mag)
    motion_variance = np.var(mag)
    chaos_score = np.std(ang) if len(ang) > 0 else 0.0
    return motion_magnitude, motion_variance, chaos_score, flow

# ── Visual Heuristics Fallbacks ─────────────────────────
def detect_visual_anomalies_heuristics(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - 10)
    cy1 = max(0, y1 - 10)
    cx2 = min(w, x2 + 10)
    cy2 = min(h, y2 + 10)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False, False, False
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 100, 100], dtype=np.uint8)
    upper_fire = np.array([20, 255, 255], dtype=np.uint8)
    mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_ratio = np.sum(mask_fire > 0) / mask_fire.size
    has_fire = fire_ratio > 0.02
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_crop)
    lower_smoke = np.array([0, 0, 50], dtype=np.uint8)
    upper_smoke = np.array([180, 50, 200], dtype=np.uint8)
    mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
    smoke_ratio = np.sum(mask_smoke > 0) / mask_smoke.size
    has_smoke = (std_dev < 15) and (smoke_ratio > 0.05)
    
    return has_fire, has_smoke, False

def detect_visual_anomalies_yolo(frame, x1, y1, x2, y2, fire_smoke_model):
    if fire_smoke_model is None:
        return detect_visual_anomalies_heuristics(frame, x1, y1, x2, y2)
        
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - 20)
    cy1 = max(0, y1 - 20)
    cx2 = min(w, x2 + 20)
    cy2 = min(h, y2 + 20)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False, False, False
        
    results = fire_smoke_model(crop, verbose=False)
    has_fire = False
    has_smoke = False
    has_debris = False
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if conf > 0.35:
                    cls_name = fire_smoke_model.names[cls_id].lower()
                    if "fire" in cls_name:
                        has_fire = True
                    elif "smoke" in cls_name:
                        has_smoke = True
                    elif "debris" in cls_name or "part" in cls_name or "cargo" in cls_name:
                        has_debris = True
                        
    return has_fire, has_smoke, has_debris

# ── Explainable Pertribution Attribution ────────────────
def explain_mlp_prediction(mlp_model, feature_tensor, scaler_mean, scaler_std, confidence):
    if mlp_model is None:
        return {}
    feature_names = [
        "YOLO Deep Anomaly", "LSTM Temporal Anomaly", "Deceleration Shock", "Aspect Ratio Shock",
        "Area Compression", "Kinetic Motion Spike", "Post-Impact Standstill", "Interaction Proximity",
        "Trajectory Deviation", "Spatiotemporal Persistence", "High Jerk Event", "Dangerous TTC",
        "Wrong-Way Motion", "Sudden Lane Departure", "Vehicle Spin", "Fire Indicator", "Smoke Indicator",
        "Debris/Damage Indicator", "Optical Flow Magnitude", "Optical Flow Variance", "Optical Flow Chaos",
        "Average Vehicle Speed", "Maximum Jerk", "Minimum TTC"
    ]
    n_features = len(feature_names)
    # Create a batch of perturbed feature vectors
    perturbed_batch = feature_tensor.repeat(n_features, 1)
    # Zero out the diagonal elements (perturb each feature to 0.0)
    for i in range(n_features):
        perturbed_batch[i, i] = 0.0
        
    with torch.no_grad():
        logits = mlp_model(perturbed_batch)
        probs = torch.softmax(logits, dim=1)
        perturbed_confs = probs[:, 1].cpu().numpy()
        
    attributions = [max(0.0, float(confidence - conf)) for conf in perturbed_confs]
    
    total_attr = sum(attributions)
    explanations = {}
    if total_attr > 0:
        for idx, name in enumerate(feature_names):
            pct = (attributions[idx] / total_attr) * 100
            if pct > 4.0:
                explanations[name] = float(round(pct, 1))
    return explanations

# ── Accident State Machine ──────────────────────────────
class AccidentStateMachine:
    def __init__(self, fps=25.0):
        self.state = "NORMAL"
        self.fps = fps
        self.cooldown_remaining = 0
        self.collision_pair = None
        
    def update(self, confidence, has_jerk, has_ttc, stopped_after_impact, collision_pair):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            
        if self.state == "NORMAL":
            if confidence >= 0.70:
                self.state = "COLLISION"
                self.collision_pair = collision_pair
                self.cooldown_remaining = int(5.0 * self.fps)
            elif has_jerk or has_ttc:
                self.state = "RISK"
                
        elif self.state == "RISK":
            if confidence >= 0.70:
                self.state = "COLLISION"
                self.collision_pair = collision_pair
                self.cooldown_remaining = int(5.0 * self.fps)
            elif not has_jerk and not has_ttc:
                self.state = "NORMAL"
                
        elif self.state == "COLLISION":
            if stopped_after_impact:
                self.state = "POST_IMPACT"
                self.cooldown_remaining = int(5.0 * self.fps)
            elif confidence < 0.40 and self.cooldown_remaining <= 0:
                self.state = "RECOVERY"
                self.cooldown_remaining = int(3.0 * self.fps)
                
        elif self.state == "POST_IMPACT":
            if confidence < 0.40 and self.cooldown_remaining <= 0:
                self.state = "RECOVERY"
                self.cooldown_remaining = int(3.0 * self.fps)
            else:
                self.state = "POST_IMPACT"
                
        elif self.state == "RECOVERY":
            if confidence >= 0.70:
                self.state = "COLLISION"
                self.collision_pair = collision_pair
                self.cooldown_remaining = int(5.0 * self.fps)
            elif self.cooldown_remaining <= 0:
                self.state = "NORMAL"
                self.collision_pair = None
                
        return self.state, self.collision_pair

# ── Core Feature Extraction and Inference Engine ────────
def run_inference(frame, buffer, yolo, lstm, vehicle_tracks=None, accident_conf_val=ACCIDENT_CONF, lstm_threshold_val=LSTM_THRESHOLD, state_machine=None, last_small_gray_container=None):
    if vehicle_tracks is None:
        vehicle_tracks = {}
        
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = frame_transform(rgb)

    # 1. Deep learned models prediction
    yolo_results = yolo(frame, verbose=False)
    top_cls = yolo_results[0].probs.top1
    top_conf = yolo_results[0].probs.top1conf.item()
    yolo_acc = top_conf if top_cls == 0 else (1 - top_conf)

    # Optimize feature extraction: extract feature for the new frame only and buffer it
    with torch.no_grad():
        feature_single = lstm.extractor(tensor.unsqueeze(0).unsqueeze(0).to(get_device()))
    buffer.append(feature_single)

    lstm_acc = 0.0
    if len(buffer) == SEQUENCE_LEN:
        feat_seq = torch.cat(list(buffer), dim=1).to(get_device())
        with torch.no_grad():
            out, _ = lstm.lstm(feat_seq)
            last   = out[:, -1, :]
            mean_p = out.mean(dim=1)
            max_p  = out.max(dim=1)[0]
            combined = torch.cat([last, mean_p, max_p], dim=1)
            logits = lstm.classifier(combined)
            probs = torch.softmax(logits, dim=1)[0]
        lstm_acc = probs[0].item()

    # 2. Road Lane Detection
    lane_angle = detect_lanes(frame)

    # 3. Dense Optical Flow with FFT Camera Motion Compensation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (160, 120))
    motion_mag, motion_var, chaos_score = 0.0, 0.0, 0.0
    flow = None
    
    if last_small_gray_container is not None:
        prev_gray = last_small_gray_container[0]
        if prev_gray is not None:
            dx, dy = estimate_camera_motion(prev_gray, small_gray)
            motion_mag, motion_var, chaos_score, flow = compute_compensated_flow(prev_gray, small_gray, dx, dy)
        last_small_gray_container[0] = small_gray
    else:
        last_small_gray_container = [small_gray]

    # 4. Vehicle Tracking & Dynamic Telemetry (ByteTrack-based logic)
    detector = load_vehicle_detector()
    cls_ids = [2, 3, 5, 7]
    det_results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=cls_ids, imgsz=480)
    boxes = det_results[0].boxes
    cls_names = {2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}

    current_tracks = {}
    box_to_track_id = {}
    detected_boxes = []
    
    if boxes is not None and len(boxes):
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            name = cls_names.get(cls_id, "Vehicle")
            
            track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else None
            if track_id is None:
                continue

            box_to_track_id[box_idx] = track_id
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            ar = (x2 - x1) / float(y2 - y1 + 1e-6)

            detected_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "name": name, "track_id": track_id, "conf": conf
            })

            if track_id not in vehicle_tracks:
                vehicle_tracks[track_id] = {
                    "centroids": [], "areas": [], "ars": [], "velocities": [],
                    "accelerations": [], "jerks": [], "directions": [], "frames_tracked": 0
                }

            history = vehicle_tracks[track_id]
            history["centroids"].append((cx, cy))
            history["areas"].append(area)
            history["ars"].append(ar)
            history["frames_tracked"] += 1

            if len(history["centroids"]) >= 2:
                last_cx, last_cy = history["centroids"][-2]
                v = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                history["velocities"].append(v)
                angle = np.arctan2(cy - last_cy, cx - last_cx) * 180 / np.pi
                history["directions"].append(angle)
            else:
                history["velocities"].append(0.0)
                history["directions"].append(0.0)

            acc = history["velocities"][-1] - history["velocities"][-2] if len(history["velocities"]) >= 2 else 0.0
            history["accelerations"].append(acc)
            jerk = history["accelerations"][-1] - history["accelerations"][-2] if len(history["accelerations"]) >= 2 else 0.0
            history["jerks"].append(jerk)

            for k in ["centroids", "areas", "ars", "velocities", "accelerations", "jerks", "directions"]:
                if len(history[k]) > 60:
                    history[k] = history[k][-60:]
            current_tracks[track_id] = history

    kinetic_spike = motion_mag > 2.5
    has_vehicles = len(current_tracks) >= 1
    deceleration_shock = False
    aspect_distortion = False
    area_compress = False
    standstill = False
    direction_deviation = False
    persistence = False
    close_proximity = False
    high_jerk_event = False
    dangerous_ttc = False
    wrong_way_motion = False
    sudden_lane_departure = False
    vehicle_spin = False
    has_fire = False
    has_smoke = False
    has_debris = False
    
    # 5. Scene Understanding Visual Anomalies (Fire/Smoke/Debris)
    fire_smoke_model = load_fire_smoke_detector()
    if fire_smoke_model is not None:
        try:
            results = fire_smoke_model(frame, verbose=False, imgsz=480)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if conf > 0.35:
                        cls_name = fire_smoke_model.names[cls_id].lower()
                        if "fire" in cls_name:
                            has_fire = True
                        elif "smoke" in cls_name:
                            has_smoke = True
                        elif "debris" in cls_name or "part" in cls_name or "cargo" in cls_name or "damage" in cls_name:
                            has_debris = True
        except Exception as e:
            print(f"Error in fire_smoke_model inference: {e}")
    else:
        # Fallback to HSV heuristics on vehicle crops
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].tolist())
                f_fire, f_smoke, _ = detect_visual_anomalies_heuristics(frame, x1, y1, x2, y2)
                if f_fire: has_fire = True
                if f_smoke: has_smoke = True

    for track_id, history in current_tracks.items():
        if history["frames_tracked"] >= 5:
            persistence = True
        if len(history["velocities"]) >= 2:
            last_v = history["velocities"][-2]
            curr_v = history["velocities"][-1]
            if last_v > 4.0 and curr_v < last_v * 0.50:
                deceleration_shock = True
        if len(history["ars"]) >= 2:
            last_ar = history["ars"][-2]
            curr_ar = history["ars"][-1]
            if abs(curr_ar - last_ar) / (last_ar + 1e-6) > 0.20:
                aspect_distortion = True
        if len(history["areas"]) >= 2:
            last_area = history["areas"][-2]
            curr_area = history["areas"][-1]
            if abs(curr_area - last_area) / (last_area + 1e-6) > 0.25:
                area_compress = True
        if len(history["velocities"]) >= 3:
            recent_v = history["velocities"][-3:-1]
            curr_v = history["velocities"][-1]
            if any(v > 3.0 for v in recent_v) and curr_v < 1.0:
                standstill = True
        if len(history["directions"]) >= 2:
            last_dir = history["directions"][-2]
            curr_dir = history["directions"][-1]
            diff_dir = abs(curr_dir - last_dir)
            if diff_dir > 180:
                diff_dir = 360 - diff_dir
            if diff_dir > 40.0:
                direction_deviation = True
                vehicle_spin = True

        if len(history["jerks"]) >= 1:
            curr_jerk = abs(history["jerks"][-1])
            if curr_jerk > 6.0:
                high_jerk_event = True

        if len(history["directions"]) >= 1:
            h_angle = history["directions"][-1]
            if abs(h_angle - lane_angle) > 135.0 and abs(h_angle - lane_angle) < 225.0:
                wrong_way_motion = True
            if abs(h_angle - lane_angle) > 45.0 and abs(h_angle - lane_angle) < 135.0:
                sudden_lane_departure = True

    collision_pair = None
    if boxes is not None and len(boxes) >= 2:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                track_id1 = box_to_track_id.get(i)
                track_id2 = box_to_track_id.get(j)
                if track_id1 is None or track_id2 is None:
                    continue
                    
                hist1 = current_tracks.get(track_id1)
                hist2 = current_tracks.get(track_id2)
                if hist1 and hist2 and len(hist1["velocities"]) >= 1 and len(hist2["velocities"]) >= 1:
                    c1 = np.array(hist1["centroids"][-1])
                    c2 = np.array(hist2["centroids"][-1])
                    dist = np.linalg.norm(c1 - c2)
                    
                    x1_1, y1_1, x2_1, y2_1 = map(int, boxes[i].xyxy[0].tolist())
                    x1_2, y1_2, x2_2, y2_2 = map(int, boxes[j].xyxy[0].tolist())
                    
                    xA = max(x1_1, x1_2)
                    yA = max(y1_1, y1_2)
                    xB = min(x2_1, x2_2)
                    yB = min(y2_1, y2_2)
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    if interArea > 0 or dist < ((x2_1 - x1_1) + (x2_2 - x1_2)) * 0.6:
                        close_proximity = True
                        
                    dir_vec = (c2 - c1) / (dist + 1e-6)
                    rad1 = hist1["directions"][-1] * np.pi / 180.0
                    rad2 = hist2["directions"][-1] * np.pi / 180.0
                    v1_vec = hist1["velocities"][-1] * np.array([np.cos(rad1), np.sin(rad1)])
                    v2_vec = hist2["velocities"][-1] * np.array([np.cos(rad2), np.sin(rad2)])
                    
                    v_rel = v1_vec - v2_vec
                    v_closing = np.dot(v_rel, dir_vec)
                    
                    if v_closing > 0.8:
                        ttc = dist / v_closing
                        if ttc < 6.0:
                            dangerous_ttc = True
                            collision_pair = (track_id1, track_id2)
    else:
        if boxes is not None and len(boxes) == 1 and yolo_acc >= accident_conf_val:
            close_proximity = True

    avg_speed = np.mean([h["velocities"][-1] for h in current_tracks.values()]) if current_tracks else 0.0
    max_jerk = np.max([abs(h["jerks"][-1]) for h in current_tracks.values()]) if current_tracks else 0.0
    min_ttc = 60.0 if not dangerous_ttc else 3.0
    
    raw_feature_vector = np.array([
        yolo_acc, lstm_acc,
        float(deceleration_shock), float(aspect_distortion), float(area_compress),
        float(kinetic_spike), float(standstill), float(close_proximity),
        float(direction_deviation), float(persistence), float(high_jerk_event),
        float(dangerous_ttc), float(wrong_way_motion), float(sudden_lane_departure),
        float(vehicle_spin), float(has_fire), float(has_smoke), float(has_debris),
        motion_mag, motion_var, chaos_score, avg_speed, max_jerk, min_ttc
    ], dtype=np.float32)

    mlp_model, scaler = load_mlp_model()
    confidence = 0.0
    feature_tensor = torch.from_numpy(raw_feature_vector).unsqueeze(0).to(get_device())
    
    if mlp_model is not None and scaler is not None:
        try:
            mean = np.array(scaler["mean"], dtype=np.float32)
            std = np.array(scaler["std"], dtype=np.float32)
            norm_feature_vector = (raw_feature_vector - mean) / (std + 1e-6)
            feature_tensor = torch.from_numpy(norm_feature_vector).unsqueeze(0).to(get_device())
            with torch.no_grad():
                logits = mlp_model(feature_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                confidence = probs[1].item()
        except Exception as e:
            print(f"MLP Inference failed: {e}")
            mlp_model = None

    if mlp_model is None:
        weights = {
            "YOLO Deep Anomaly": 0.25, "LSTM Temporal Anomaly": 0.20, "Deceleration Shock": 0.08,
            "Aspect Ratio Shock": 0.07, "Area Compression": 0.05, "Kinetic Motion Spike": 0.05,
            "Post-Impact Standstill": 0.05, "Interaction Proximity": 0.10, "High Jerk Event": 0.10,
            "Dangerous TTC": 0.10, "Optical Flow Chaos": 0.05
        }
        active_weights = 0.0
        if yolo_acc >= accident_conf_val: active_weights += weights["YOLO Deep Anomaly"]
        if lstm_acc >= lstm_threshold_val: active_weights += weights["LSTM Temporal Anomaly"]
        if deceleration_shock: active_weights += weights["Deceleration Shock"]
        if aspect_distortion: active_weights += weights["Aspect Ratio Shock"]
        if area_compress: active_weights += weights["Area Compression"]
        if kinetic_spike: active_weights += weights["Kinetic Motion Spike"]
        if standstill: active_weights += weights["Post-Impact Standstill"]
        if close_proximity: active_weights += weights["Interaction Proximity"]
        if high_jerk_event: active_weights += weights["High Jerk Event"]
        if dangerous_ttc: active_weights += weights["Dangerous TTC"]
        if chaos_score > 1.2: active_weights += weights["Optical Flow Chaos"]
        confidence = active_weights

    features = {
        "YOLO Deep Anomaly": yolo_acc >= accident_conf_val,
        "LSTM Temporal Anomaly": lstm_acc >= lstm_threshold_val,
        "Vehicle Presence": has_vehicles,
        "Deceleration Shock": deceleration_shock,
        "Aspect Ratio Shock": aspect_distortion,
        "Area Compression": area_compress,
        "Kinetic Motion Spike": kinetic_spike,
        "Post-Impact Standstill": standstill,
        "Interaction Proximity": close_proximity,
        "Trajectory Deviation": direction_deviation,
        "Spatiotemporal Persistence": persistence,
        "High Jerk Event": high_jerk_event,
        "Dangerous TTC": dangerous_ttc,
        "Wrong-Way Motion": wrong_way_motion,
        "Sudden Lane Departure": sudden_lane_departure,
        "Vehicle Spin": vehicle_spin,
        "Fire Indicator": has_fire,
        "Smoke Indicator": has_smoke,
        "Debris Indicator": has_debris
    }

    mean_val = scaler["mean"] if scaler else [0.0]*24
    std_val = scaler["std"] if scaler else [1.0]*24
    explanations = explain_mlp_prediction(mlp_model, feature_tensor, mean_val, std_val, confidence)

    if state_machine is None:
        state_machine = AccidentStateMachine()
        
    alert_state, matched_pair = state_machine.update(
        confidence, high_jerk_event, dangerous_ttc, standstill, collision_pair
    )
    
    st.session_state.last_features = features
    
    # Draw bounding boxes (highlight accident with red boundary square)
    for db in detected_boxes:
        x1, y1, x2, y2 = db["x1"], db["y1"], db["x2"], db["y2"]
        name = db["name"]
        track_id = db["track_id"]
        conf = db["conf"]
        
        is_accident_vehicle = False
        if matched_pair is not None and track_id in matched_pair:
            is_accident_vehicle = True
        elif alert_state in ["COLLISION", "POST_IMPACT", "Confirmed Accident"]:
            # Check if this specific vehicle had anomalies (e.g. standstill, shock, spin, high jerk)
            # Or if it's the only vehicle in a critical threat situation
            history = current_tracks.get(track_id)
            if history:
                has_decel = False
                if len(history["velocities"]) >= 2:
                    last_v = history["velocities"][-2]
                    curr_v = history["velocities"][-1]
                    if last_v > 4.0 and curr_v < last_v * 0.50:
                        has_decel = True
                
                has_standstill = False
                if len(history["velocities"]) >= 3:
                    recent_v = history["velocities"][-3:-1]
                    curr_v = history["velocities"][-1]
                    if any(v > 3.0 for v in recent_v) and curr_v < 1.0:
                        has_standstill = True

                has_high_jerk = False
                if len(history["jerks"]) >= 1:
                    if abs(history["jerks"][-1]) > 6.0:
                        has_high_jerk = True
                
                if has_decel or has_standstill or has_high_jerk or len(detected_boxes) <= 2:
                    is_accident_vehicle = True
            else:
                is_accident_vehicle = True
                
        if is_accident_vehicle:
            # Thick red warning boundary square
            color = (0, 0, 255) # BGR Red
            thickness = 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"💥 ACCIDENT VEHICLE #{track_id}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Standard yellow box
            color = (0, 255, 255) # BGR Yellow
            thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{name} #{track_id} {conf:.0%}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    meta_info = {
        "confidence": confidence,
        "alert_state": alert_state,
        "explanations": explanations,
        "collision_pair": matched_pair,
        "vehicle_speeds": {tid: round(h["velocities"][-1], 1) for tid, h in current_tracks.items()},
        "raw_features": raw_feature_vector.tolist()
    }

    return alert_state, yolo_acc, lstm_acc, frame, meta_info
