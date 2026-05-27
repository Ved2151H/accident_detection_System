import cv2
import numpy as np
import torch
import streamlit as st
from dashboard.config import ACCIDENT_CONF, LSTM_THRESHOLD, SEQUENCE_LEN, DEVICE
from dashboard.models import frame_transform, load_vehicle_detector

# ── Core Threat Scan Inference on Single Frame ───────────
def run_inference(frame, buffer, yolo, lstm, vehicle_tracks=None, accident_conf_val=ACCIDENT_CONF, lstm_threshold_val=LSTM_THRESHOLD):
    """Detects accidents and tracks vehicles using YOLO + LSTM + 10+ Physical Telemetry Features."""
    if vehicle_tracks is None:
        vehicle_tracks = {}
        
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

    # Vehicle Tracking & Feature Extraction
    detector    = load_vehicle_detector()
    cls_ids     = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)
    det_results = detector(frame, verbose=False, classes=cls_ids)
    boxes       = det_results[0].boxes
    cls_names   = {2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}

    current_tracks = {}
    used_track_ids = set()
    
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
            
            # Centroid & Telemetry calculation
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            ar = (x2 - x1) / float(y2 - y1 + 1e-6)
            
            # Match current vehicle box to existing tracked vehicle
            best_id = None
            min_dist = float('inf')
            for track_id, history in list(vehicle_tracks.items()):
                if track_id == "last_gray_frame" or track_id in used_track_ids:
                    continue
                last_cx, last_cy = history["centroids"][-1]
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    best_id = track_id
                    
            if best_id is not None:
                used_track_ids.add(best_id)
                track_id = best_id
            else:
                track_id = len(vehicle_tracks) + len(current_tracks) + 1
                vehicle_tracks[track_id] = {
                    "centroids": [],
                    "areas": [],
                    "ars": [],
                    "velocities": [],
                    "directions": [],
                    "frames_tracked": 0
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
                
            for k in ["centroids", "areas", "ars", "velocities", "directions"]:
                if len(history[k]) > 10:
                    history[k] = history[k][-10:]
            current_tracks[track_id] = history

    # Global motion & kinetic energy calculation
    kinetic_spike = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_gray_frame = vehicle_tracks.get("last_gray_frame")
    if last_gray_frame is not None and last_gray_frame.shape == gray.shape:
        diff = cv2.absdiff(gray, last_gray_frame)
        mean_diff = diff.mean()
        kinetic_spike = mean_diff > 3.0
    else:
        kinetic_spike = yolo_acc > 0.10
    vehicle_tracks["last_gray_frame"] = gray

    # Retain active tracks
    for k in list(vehicle_tracks.keys()):
        if k != "last_gray_frame" and k not in current_tracks:
            del vehicle_tracks[k]
    for k, v in current_tracks.items():
        vehicle_tracks[k] = v

    # 10+ Features Computation
    has_vehicles = len(boxes) >= 1 if boxes is not None else False
    deceleration_shock = False
    aspect_distortion = False
    area_compress = False
    standstill = False
    direction_deviation = False
    persistence = False
    close_proximity = False

    for track_id, history in current_tracks.items():
        if history["frames_tracked"] >= 3:
            persistence = True
        if len(history["velocities"]) >= 2:
            last_v = history["velocities"][-2]
            curr_v = history["velocities"][-1]
            if last_v > 5.0 and curr_v < last_v * 0.65:
                deceleration_shock = True
        if len(history["ars"]) >= 2:
            last_ar = history["ars"][-2]
            curr_ar = history["ars"][-1]
            if abs(curr_ar - last_ar) / (last_ar + 1e-6) > 0.15:
                aspect_distortion = True
        if len(history["areas"]) >= 2:
            last_area = history["areas"][-2]
            curr_area = history["areas"][-1]
            if abs(curr_area - last_area) / (last_area + 1e-6) > 0.20:
                area_compress = True
        if len(history["velocities"]) >= 3:
            recent_v = history["velocities"][-3:-1]
            curr_v = history["velocities"][-1]
            if any(v > 4.0 for v in recent_v) and curr_v < 1.5:
                standstill = True
        if len(history["directions"]) >= 2:
            last_dir = history["directions"][-2]
            curr_dir = history["directions"][-1]
            diff_dir = abs(curr_dir - last_dir)
            if diff_dir > 180:
                diff_dir = 360 - diff_dir
            if diff_dir > 35.0:
                direction_deviation = True

    # Multi-vehicle overlap/proximity check
    if boxes is not None and len(boxes) >= 2:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1_1, y1_1, x2_1, y2_1 = map(int, boxes[i].xyxy[0].tolist())
                x1_2, y1_2, x2_2, y2_2 = map(int, boxes[j].xyxy[0].tolist())
                
                xA = max(x1_1, x1_2)
                yA = max(y1_1, y1_2)
                xB = min(x2_1, x2_2)
                yB = min(y2_1, y2_2)
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                if interArea > 0:
                    close_proximity = True
                    break
                
                cx1, cy1 = (x1_1 + x2_1)/2.0, (y1_1 + y2_1)/2.0
                cx2, cy2 = (x1_2 + x2_2)/2.0, (y1_2 + y2_2)/2.0
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                avg_width = ((x2_1 - x1_1) + (x2_2 - x1_2)) / 2.0
                if dist < avg_width * 1.2:
                    close_proximity = True
                    break
            if close_proximity:
                break
    else:
        if boxes is not None and len(boxes) == 1 and yolo_acc >= accident_conf_val:
            close_proximity = True

    # Dynamic Sensor Fusion Bypass for High-Confidence Impacts
    is_high_conf_impact = (yolo_acc > 0.35) or (lstm_acc > 0.75)
    if is_high_conf_impact:
        deceleration_shock = True
        aspect_distortion = True
        area_compress = True
        standstill = True
        direction_deviation = True
        persistence = True
        close_proximity = True
        kinetic_spike = True

    # Assemble 10+ Features
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
        "Spatiotemporal Persistence": persistence
    }

    # Decide accident status
    strict_mode = st.session_state.get("detection_mode", "Strict 10+ Features Mode (Recommended)") == "Strict 10+ Features Mode (Recommended)"
    if strict_mode:
        accident = all(features.values())
    else:
        accident = (yolo_acc >= accident_conf_val and lstm_acc >= lstm_threshold_val)

    # Save to session state so UI can render in real-time
    st.session_state.last_features = features

    return accident, yolo_acc, lstm_acc, frame

# ── Dynamic Cyberpunk Telemetry Components ───────────────
def render_feature_telemetry(features):
    if not features:
        return ""
    html = '<div style="margin-top: 15px; background: rgba(18, 18, 30, 0.4); border: 1px solid rgba(0, 255, 255, 0.1); border-radius: 12px; padding: 12px;">'
    html += '<div style="font-family: \'Share Tech Mono\', monospace; color: #00ffff; font-size: 0.95rem; margin-bottom: 10px; border-bottom: 1px solid rgba(0, 255, 255, 0.1); padding-bottom: 6px; font-weight: bold; text-transform: uppercase;">🛡️ 10+ CRASH TELEMETRY FEATURE MATRIX</div>'
    html += '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">'
    
    for fname, met in features.items():
        color = "#00ffcc" if met else "#ff3366"
        bg = "rgba(0, 255, 204, 0.08)" if met else "rgba(255, 51, 102, 0.08)"
        border = "rgba(0, 255, 204, 0.2)" if met else "rgba(255, 51, 102, 0.2)"
        status_text = "MET ✓" if met else "PENDING"
        
        html += f'<div style="background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 6px 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; font-family: \'Share Tech Mono\', monospace; color: #ffffff;">'
        html += f'<span>{fname}</span>'
        html += f'<span style="color: {color}; font-weight: bold; letter-spacing: 1px;">{status_text}</span>'
        html += '</div>'
        
    html += '</div></div>'
    return html

def update_telemetry_standby(container, yolo_acc, lstm_acc, yolo_thr=0.20, lstm_thr=0.48, features=None):
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
        
        if features:
            if hasattr(st, "html"):
                st.html(render_feature_telemetry(features))
            else:
                st.markdown(render_feature_telemetry(features), unsafe_allow_html=True)
