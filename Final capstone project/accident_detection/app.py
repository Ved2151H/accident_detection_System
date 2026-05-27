"""
👀 AEGIS EYE — Accident Localization & Dispatch Dashboard
Modular runner file acting as clean entrypoint coordinator.
Run: streamlit run app.py
"""

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

# ── Import Modular Components from dashboard package ──────
from dashboard.config import (
    YOLO_WEIGHTS, LSTM_WEIGHTS, DB_PATH, SNAPSHOTS_DIR,
    SEQUENCE_LEN, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS,
    NUM_CLASSES, FRAME_SKIP, ACCIDENT_CONF, LSTM_THRESHOLD,
    DEVICE, AdaptiveThreatCalibrator
)
from dashboard.database import (
    init_db, log_incident, get_incidents, save_snapshot
)
from dashboard.map import get_leaflet_html
from dashboard.models import (
    load_models, load_vehicle_detector, frame_transform
)
from dashboard.telemetry import (
    run_inference, update_telemetry_standby, render_feature_telemetry
)
from dashboard.utils import (
    generate_gps, get_camera_location, randomize_camera_location, digipin_helper
)
def render_html(html_str, element=st):
    # lstrip each line to prevent markdown preformatted code-block conversion
    clean_lines = [line.lstrip() for line in html_str.splitlines()]
    clean_html = "\n".join(clean_lines)
    if hasattr(st, "html"):
        element.html(clean_html)
    else:
        element.markdown(clean_html, unsafe_allow_html=True)

st.set_page_config(page_title="Accident Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "Cyberpunk Dark"

# Theme Selection in Sidebar (renders at top of sidebar)
with st.sidebar:
    theme_choice = st.radio("🎨 Interface Theme Mode", ["Cyberpunk Dark", "Premium Light"])
    st.session_state.theme = theme_choice

def load_css(theme):
    css_file = Path("dashboard") / ("dark.css" if theme == "Cyberpunk Dark" else "light.css")
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(st.session_state.theme)

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
if "detection_mode" not in st.session_state:
    st.session_state.detection_mode = "Strict 10+ Features Mode (Recommended)"
if "last_features" not in st.session_state:
    st.session_state.last_features = {}

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
    render_html("""
    <div style='text-align: center; margin-top: 15px;'>
        <h1 class='sidebar-title'>👀 AEGIS EYE</h1>
        <div class='sidebar-subtitle'>ACCIDENT LOCALIZATION & DISPATCH</div>
    </div>
    """)

    st.divider()
    # Navigation removed per user request

    # Source Selection
    source_type = st.selectbox("Select Camera Input Stream", ["📷 Laptop / USB Webcam", "📁 High-Res Demo footage", "📤 Upload Custom Video", "📡 CCTV RTSP Stream"])
    video_source = None
    source_name = ""
    source_key = "default"

    if source_type == "📷 Laptop / USB Webcam":
        cam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
        video_source = cam_index
        source_name = f"Webcam {cam_index}"
        source_key = f"webcam_{cam_index}"
    elif source_type == "📁 High-Res Demo footage":
        accident_dir = Path("data/raw/accident")
        video_files = []
        if accident_dir.exists():
            video_files = list(accident_dir.glob("*.mp4")) + list(accident_dir.glob("*.avi"))
        
        if video_files:
            selected_file = st.selectbox("Select Demo Video Clip", [f.name for f in video_files])
            video_source = str(accident_dir / selected_file)
            source_name = f"Demo clip: {selected_file}"
            source_key = f"demo_{selected_file}"
        else:
            st.info("No demo clips found in data/raw/accident. You can upload a file instead using the 'Upload Custom Video' option.")
            source_key = "demo_none"
    elif source_type == "📤 Upload Custom Video":
        uploaded = st.file_uploader("Browse and upload video file", type=["mp4", "avi"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
            tfile.write(uploaded.read())
            tfile.close()
            video_source = tfile.name
            source_name = f"Upload: {uploaded.name}"
            source_key = f"upload_{uploaded.name}"
            st.success(f"Video uploaded successfully: {uploaded.name}")
            st.session_state.video_uploaded = True
        else:
            source_key = "upload_pending"
            st.session_state.video_uploaded = False
    else:
        video_source = st.text_input("Enter RTSP stream address", "rtsp://192.168.1.100:554/stream")
        source_name = "RTSP Network Stream"
        source_key = f"rtsp_{video_source}"

    # Persistent camera location display & controls
    loc_data = get_camera_location(source_key)
    st.markdown('<div style="font-family: \'Share Tech Mono\', monospace; color: #0d9488; font-weight: bold; margin-bottom: 4px; margin-top: 8px;">📍 CAMERA GEOLOCATION (DUMMY)</div>', unsafe_allow_html=True)
    
    loc_card_border = "rgba(0, 255, 255, 0.2)" if st.session_state.theme == "Cyberpunk Dark" else "rgba(13, 148, 136, 0.2)"
    loc_text_color = "#00ffff" if st.session_state.theme == "Cyberpunk Dark" else "#0d9488"
    render_html(f"""
    <div class="telemetry-card" style="border-color: {loc_card_border}; margin-bottom: 12px; padding: 12px;">
        <span style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #888;">Assigned Spot</span>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 1rem; font-weight: bold; color: {loc_text_color}; margin-top: 4px;">📍 {loc_data['city_name']}</div>
        <div style="font-size: 0.8rem; margin-top: 6px; line-height: 1.4;">
            <b>LATITUDE:</b> {loc_data['lat']:.6f}° N<br>
            <b>LONGITUDE:</b> {loc_data['lon']:.6f}° E
        </div>
        <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
            <b>INDIA POST DIGIPIN:</b>
            <span style="display: block; font-family: 'Share Tech Mono', monospace; font-weight: 800; font-size: 0.95rem; color: {loc_text_color}; margin-top: 2px;">🇮🇳 {loc_data['digipin']}</span>
        </div>
    </div>
    """)
    
    if st.button("🔄 Re-randomize Camera Location", width="stretch"):
        randomize_camera_location(source_key)
        st.rerun()

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
        accident_conf = st.slider("YOLO Accident Threshold", min_value=0.05, max_value=0.95, value=0.20, step=0.01, help="Calibrate YOLO classification trigger level.")
        lstm_threshold = st.slider("LSTM Anomaly Threshold", min_value=0.05, max_value=0.95, value=0.65, step=0.01, help="Calibrate sequential temporal trigger level.")

    st.divider()
    st.markdown("**🛡️ DETECTION RIGOR MODE**")
    detection_mode = st.radio(
        "Select Detection Mode",
        ["Strict 10+ Features Mode (Recommended)", "Standard YOLO+LSTM Mode"],
        help="Strict Mode requires all 10+ visual/physical telemetry features to be satisfied to trigger an accident."
    )
    st.session_state.detection_mode = detection_mode
    
    st.divider()

    # System Buttons
    col_start, col_stop = st.columns(2)
    start_sys = col_start.button("▶ START SYSTEM", type="primary", width="stretch")
    stop_sys = col_stop.button("⏹ STOP SYSTEM", width="stretch")

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

# ════════════════════════════════════════════════════════
# HEADER BANNER RENDER
# ════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align: center; padding: 10px 0; margin-bottom: 25px;'>
    <h2 style='font-family: "Outfit", sans-serif; font-weight: 800; font-size: 2.2rem; color: #ffffff; letter-spacing: 2px; margin: 0; text-transform: uppercase;'>
        Road accident detection system
    </h2>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# MAIN SCREEN LAYOUT
# ════════════════════════════════════════════════════════
left_pane, right_pane = st.columns([0.65, 0.35])
alert_container = st.empty()  # Alert placeholder

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
        video_container.image(st.session_state.frozen_frame, width="stretch")
    else:
        video_container.warning("Incident snapshot frame not found on disk.")

    with status_container:
        data = st.session_state.current_incident_data
        render_html(f"""
        <div class="accident-banner-red">
            <h3 style="margin: 0; color: white;">🚨 COLLISION DETECTED & SYSTEM FROZEN</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                An active vehicle accident threat was detected on <b>{data['source']}</b> at <b>{data['timestamp']}</b>.
                The camera feed has been frozen at the anomaly timestamp. Automatic rescue dispatch localizing.
            </p>
        </div>
        """)
        # Use DigiPin-based Leaflet map for incident location
        if st.session_state.current_coords:
            lat, lon, digipin_val = st.session_state.current_coords
        else:
            loc_data = get_camera_location(source_key)
            lat, lon, digipin_val = loc_data["lat"], loc_data["lon"], loc_data["digipin"]
            
        leaflet_html = get_leaflet_html(lat, lon, digipin_val, is_incident=True)
        st.iframe(leaflet_html, height=500)
        col_m1, col_m2 = st.columns(2)
        render_html(f"""
        <div class="telemetry-card" style="border-color: rgba(255, 51, 51, 0.3);">
            <span style="color: #ff3333; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">YOLO COLLISION PROBABILITY</span>
            <div class="metric-val" style="color: #ff3333;">{data['yolo_conf']:.2%}</div>
        </div>
        """, col_m1)
        
        render_html(f"""
        <div class="telemetry-card" style="border-color: rgba(255, 51, 51, 0.3);">
            <span style="color: #ff3333; font-weight: 600; text-transform: uppercase; font-size: 0.8rem;">LSTM TEMPORAL THREAT PROBABILITY</span>
            <div class="metric-val" style="color: #ff3333;">{data['lstm_prob']:.2%}</div>
        </div>
        """, col_m2)

        if st.button("🔓 CLEAR ACTIVE ALERT & RESUME MONITORING", type="primary", width="stretch"):
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
            yolo_conf_buffer = collections.deque(maxlen=SEQUENCE_LEN)
            frame_idx = 0
            consecutive_accidents = 0
            vehicle_tracks = {}
            
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
                    frame.copy(), buffer, yolo_model, lstm_model, vehicle_tracks, accident_conf, lstm_threshold)
                yolo_conf_buffer.append(yolo_acc)
                features = st.session_state.get("last_features", {})
                
                # Update automated statistical baseline
                if auto_calib:
                    st.session_state.calibrator.update(yolo_acc, lstm_acc)
                
                # Show live video frame
                rgb_live = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_container.image(rgb_live, width="stretch")
                
                # Threat detected event trigger with consecutive frames guard
                if accident:
                    consecutive_accidents += 1
                else:
                    consecutive_accidents = 0
                
                if consecutive_accidents >= 5:
                    loc_data = get_camera_location(source_key)
                    lat, lon = loc_data["lat"], loc_data["lon"]
                    digipin_code = loc_data["digipin"]
                    
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
                update_telemetry_standby(status_container, yolo_acc, lstm_acc, accident_conf, lstm_threshold, features)
                time.sleep(0.01)
                
            cap.release()
else:
    # Standby monitoring system idle state
    video_container.info("⚙️ Threat monitoring engine standby. Select a feed input and press '▶ START SYSTEM' in the control panel to begin.")
    
    with status_container:
        render_html("""
        <div class="normal-banner-green" style="background: rgba(10, 25, 15, 0.3); border-color: #333344;">
            <h4 style="margin: 0; color: #888;">🛰️ STANDBY / AWAITING SIGNAL</h4>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #777788;">
                Inference core initialized. Ready to begin threat scanning.
            </p>
        </div>
        """)

# ════════════════════════════════════════════════════════
# GEOLOCATION DISPATCH MAP RENDERING
# ════════════════════════════════════════════════════════
# Location Alert display (Dynamic depending on state)
if st.session_state.current_coords:
    lat, lon, digipin_code = st.session_state.current_coords
    render_html(f"""
    <div style="background: linear-gradient(135deg, #7c0000 0%, #3a0000 100%); border: 2px solid #ff3333; border-radius: 12px; padding: 15px; margin-bottom: 20px; font-family: 'Share Tech Mono', monospace; box-shadow: 0 4px 15px rgba(255,51,51,0.3);">
        <h3 style="margin: 0; color: #ffffff; font-weight: bold;">🚨 EMERGENCY DISPATCH ALERT</h3>
        <p style="margin: 6px 0 0 0; font-size: 1.05rem; color: #ffcccc;">
            Accident localized at base station coordinate position:<br>
            <b>📍 LATITUDE:</b> {lat:.6f}° N &nbsp;&nbsp;|&nbsp;&nbsp; <b>📍 LONGITUDE:</b> {lon:.6f}° E<br>
            <b>🇮🇳 INDIA POST DIGIPIN:</b> {digipin_code}
        </p>
    </div>
    """, alert_container)
    
    render_html(f"""
    <div class="coord-display">
        💥 LIVE CRASH VEHICLE LOCALIZATION:<br>
        <b>LATITUDE:</b> {lat:.6f}° N<br>
        <b>LONGITUDE:</b> {lon:.6f}° E<br>
        <span style="font-size: 0.8rem; color: #ff3333; opacity: 0.8; font-weight: bold; animation: pulse 1s infinite;">
            📡 DISPATCHING GPS TRACKER BEACON TO SPOT
        </span>
    </div>
    """, coords_container)
    
    # Render premium Leaflet DigiPin map
    with map_container:
        leaflet_html = get_leaflet_html(lat, lon, digipin_code, is_incident=True, zoom=14)
        st.iframe(leaflet_html, height=350)
else:
    # Handle Location Alert when video is uploaded
    if st.session_state.get("video_uploaded", False):
        loc = get_camera_location(source_key)
        render_html(f"""
        <div style="background: rgba(13, 148, 136, 0.1); border: 2px solid #0d9488; border-radius: 12px; padding: 15px; margin-bottom: 20px; font-family: 'Share Tech Mono', monospace;">
            <h3 style="margin: 0; color: #0d9488; font-weight: bold;">🛰️ CAMERA STREAM GEOLOCATION</h3>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; color: #ffffff;">
                Active camera stream location assigned:<br>
                <b>📍 LATITUDE:</b> {loc['lat']:.6f}° N &nbsp;&nbsp;|&nbsp;&nbsp; <b>📍 LONGITUDE:</b> {loc['lon']:.6f}° E<br>
                <b>🇮🇳 INDIA POST DIGIPIN:</b> {loc['digipin']}
            </p>
        </div>
        """, alert_container)
    else:
        alert_container.empty()

    # Standby central monitoring map view (dynamic based on selected feed's location)
    loc_data = get_camera_location(source_key)
    lat_val, lon_val = loc_data["lat"], loc_data["lon"]
    digipin_val = loc_data["digipin"]
    city_val = loc_data["city_name"]
    
    border_color = "#00ffff" if st.session_state.theme == "Cyberpunk Dark" else "#0d9488"
    text_color = "#00ffff" if st.session_state.theme == "Cyberpunk Dark" else "#0f766e"
    bg_color = "#0f1620" if st.session_state.theme == "Cyberpunk Dark" else "#f1f5f9"
    
    render_html(f"""
    <div class="coord-display" style="border-left-color: {border_color}; color: {text_color}; background: {bg_color};">
        🛰️ CAMERA BASE STATION ({city_val}):<br>
        <b>LATITUDE:</b> {lat_val:.6f}° N<br>
        <b>LONGITUDE:</b> {lon_val:.6f}° E<br>
        <b>INDIA POST DIGIPIN:</b> 🇮🇳 {digipin_val}<br>
        <span style="font-size: 0.8rem; color: {border_color}; opacity: 0.8; font-weight: bold;">
            🟢 ALL PATROL BEACONS SECURE
        </span>
    </div>
    """, coords_container)
        
    with map_container:
        leaflet_html = get_leaflet_html(lat_val, lon_val, digipin_val, is_incident=False, zoom=12)
        st.iframe(leaflet_html, height=350)