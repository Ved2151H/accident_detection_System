import sys
from types import ModuleType

# Create a mock streamlit module to prevent import/runtime errors when importing dashboard modules
class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

class MockStreamlit(ModuleType):
    def cache_resource(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    def cache_data(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    def divider(self): pass
    def stop(self): sys.exit(0)

st_mock = MockStreamlit('streamlit')
st_mock.session_state = SessionState()
st_mock.session_state.theme = "Cyberpunk Dark"
st_mock.session_state.last_features = {}
st_mock.session_state.video_locations = {}
st_mock.session_state.accident_logged_this_trigger = False
sys.modules['streamlit'] = st_mock

import os
import cv2
import json
import time
import base64
import argparse
import datetime
import numpy as np
from collections import deque
from pathlib import Path

# Fix python path to allow importing from root directory
root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'backend'))

def main():
    parser = argparse.ArgumentParser(description="Aegis Eye AI Processing Worker")
    parser.add_argument("--task", type=str, required=True, choices=["collision"], help="Task to run")
    parser.add_argument("--source", type=str, required=True, help="Path to video or webcam index")
    parser.add_argument("--confidence_threshold", type=float, default=0.85, help="LSTM confidence threshold")
    parser.add_argument("--export", action="store_true", help="Export processed video")
    args = parser.parse_args()

    # Determine source value
    source_val = int(args.source) if args.source.isdigit() else args.source
    is_live = isinstance(source_val, int) or (isinstance(source_val, str) and "rtsp" in source_val.lower())

    # 1. Open the video source immediately
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(json.dumps({"type": "error", "message": f"Failed to open source: {args.source}"}), flush=True)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 2. Capture the first frame immediately for instant display
    ret, first_frame = cap.read()
    if not ret:
        print(json.dumps({"type": "error", "message": "Failed to read first frame from source"}), flush=True)
        sys.exit(1)

    h_f, w_f = first_frame.shape[:2]
    scale_f = 640.0 / w_f
    if scale_f < 1.0:
        disp_first = cv2.resize(first_frame, (640, int(h_f * scale_f)))
    else:
        disp_first = first_frame
    _, buffer_img = cv2.imencode('.jpg', disp_first, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    first_frame_b64 = base64.b64encode(buffer_img).decode('utf-8')

    # 3. Defer DB and location helper imports to keep startup ultra-fast
    from backend.database.database import init_db
    init_db()

    source_key = f"worker_{args.source}"
    from backend.utils.helpers import get_camera_location
    loc_data = get_camera_location(source_key)

    # 4. Print start event to notify frontend
    print(json.dumps({
        "type": "start",
        "task": args.task,
        "source": str(args.source),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "location": loc_data
    }), flush=True)

    # 5. Broadcast the first frame instantly to show in UI within <0.5s
    print(json.dumps({
        "type": "frame",
        "frame": f"data:image/jpeg;base64,{first_frame_b64}",
        "progress": 0.0,
        "fps": round(fps, 1),
        "frame_idx": 1,
        "alert_state": "Normal",
        "raw_prob": 0.0,
        "calibrated_prob": 0.0,
        "accident_confidence": 0.0,
        "risk_level": "Low",
        "triggered_features": [],
        "feature_explanations": {},
        "vehicle_ids": [],
        "vehicle_speeds": {},
        "collision_pair": None,
        "features": {}
    }), flush=True)

    # Set up video writer if export is enabled
    out_video = None
    output_filename = ""
    if args.export and not is_live:
        os.makedirs("logs/outputs", exist_ok=True)
        output_filename = f"{args.task}_output_{int(time.time())}.mp4"
        output_path = os.path.join("logs/outputs", output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        out_video.write(first_frame)

    frame_idx = 1
    start_time = time.time()

    if args.task == "collision":
        import threading
        from backend.utils.config import SEQUENCE_LEN, FRAME_SKIP
        
        model_container = {"yolo": None, "lstm": None, "ready": False, "error": None}
        
        def load_models_async():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(0.625, 0)
                from backend.services.detection.models import load_models
                yolo, lstm = load_models()
                if not yolo or not lstm:
                    raise RuntimeError("Failed to load weights: returned None")
                model_container["yolo"] = yolo
                model_container["lstm"] = lstm
                model_container["ready"] = True
            except Exception as e:
                model_container["error"] = str(e)
                
        load_thread = threading.Thread(target=load_models_async)
        load_thread.start()

        buffer = deque(maxlen=SEQUENCE_LEN)
        vehicle_tracks = {}
        skip_val = 1 if is_live else FRAME_SKIP
        target_interval = skip_val / fps

        # Initialize State Machine and camera motion buffer
        from backend.services.detection.telemetry import AccidentStateMachine
        state_machine = AccidentStateMachine(fps=fps)
        last_small_gray_container = [None]

        # 15-second circular buffer for event recording (5s before, 10s after)
        pre_impact_buffer_len = int(5.0 * fps)
        pre_impact_buffer = deque(maxlen=pre_impact_buffer_len)
        pre_impact_buffer.append(first_frame.copy())
        
        is_recording = False
        record_writer = None
        record_frames_remaining = 0

        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Store raw frame in pre-impact buffer
            pre_impact_buffer.append(frame.copy())

            if frame_idx % skip_val != 0:
                if out_video is not None:
                    out_video.write(frame)
                continue

            if model_container["ready"]:
                from backend.services.detection.telemetry import run_inference
                alert_state, raw_prob, lstm_prob, annotated, meta_info = run_inference(
                    frame.copy(), buffer, model_container["yolo"], model_container["lstm"], vehicle_tracks, 
                    args.confidence_threshold, state_machine=state_machine, last_small_gray_container=last_small_gray_container
                )
                
                # Map COLLISION and POST_IMPACT states to Confirmed Accident for DB and snapshot logs
                if alert_state in ["COLLISION", "POST_IMPACT"]:
                    alert_state = "Confirmed Accident"
                
                # Retrieve fusion output details
                confidence = meta_info.get("confidence", 0.0)
                explanations = meta_info.get("explanations", {})
                collision_pair = meta_info.get("collision_pair", None)
                vehicle_speeds = meta_info.get("vehicle_speeds", {})
                
                features = st_mock.session_state.get("last_features", {})
                
                # Derive Risk Level
                if confidence < 0.30:
                    risk_level = "Low"
                elif confidence < 0.55:
                    risk_level = "Medium"
                elif confidence < 0.75:
                    risk_level = "High"
                else:
                    risk_level = "Critical"
                    
                # Handle automatic pre/post impact recording
                if alert_state == "Confirmed Accident" and not is_recording:
                    is_recording = True
                    os.makedirs("logs/outputs", exist_ok=True)
                    record_filename = f"accident_record_{int(time.time())}.mp4"
                    record_path = os.path.join("logs/outputs", record_filename)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    record_writer = cv2.VideoWriter(record_path, fourcc, fps, (width, height))
                    
                    # Dump pre-impact frames
                    for pf in pre_impact_buffer:
                        record_writer.write(pf)
                    record_frames_remaining = int(10.0 * fps)
            else:
                if model_container["error"]:
                    print(json.dumps({"type": "error", "message": f"AI init failed: {model_container['error']}"}), flush=True)
                    sys.exit(1)
                    
                # Display raw frame with initializing watermark
                annotated = frame.copy()
                h_dim, w_dim = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w_dim, 50), (0, 10, 20), -1)
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                
                dots = "." * (int(time.time() * 2) % 4)
                text = f"AEGIS EYE: Initializing Threat Scan AI{dots}"
                cv2.putText(annotated, text, (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
                
                alert_state = "Normal"
                raw_prob = 0.0
                lstm_prob = 0.0
                confidence = 0.0
                risk_level = "Low"
                explanations = {}
                collision_pair = None
                vehicle_speeds = {}
                features = {}

            # Event recording continue
            if is_recording and record_writer is not None:
                record_writer.write(frame)
                record_frames_remaining -= 1
                if record_frames_remaining <= 0:
                    record_writer.release()
                    record_writer = None
                    is_recording = False
                    print(json.dumps({"type": "recording_saved", "path": record_filename}), flush=True)

            # Log incident to DB if accident confirmed
            if alert_state == "Confirmed Accident":
                if not st_mock.session_state.get("accident_logged_this_trigger", False):
                    st_mock.session_state.accident_logged_this_trigger = True
                    highlight_frame = annotated.copy()
                    h_dim, w_dim = highlight_frame.shape[:2]
                    cv2.rectangle(highlight_frame, (0, 0), (w_dim, h_dim), (0, 0, 255), 8)
                    from backend.database.database import save_snapshot, log_incident
                    snap_path = save_snapshot(highlight_frame)
                    log_incident(str(args.source), raw_prob, confidence, snap_path, loc_data["lat"], loc_data["lon"], loc_data["digipin"])
                    
                    # Print incident notification to stdout
                    print(json.dumps({
                        "type": "incident",
                        "yolo_conf": float(raw_prob),
                        "lstm_prob": float(confidence), # Map confidence to lstm_prob for compat
                        "snapshot": snap_path,
                        "lat": loc_data["lat"],
                        "lon": loc_data["lon"],
                        "digipin": loc_data["digipin"],
                        "timestamp": datetime.datetime.now().isoformat()
                    }), flush=True)
            elif alert_state == "Normal":
                st_mock.session_state.accident_logged_this_trigger = False

            # Encode annotated frame to base64
            h_dim, w_dim = annotated.shape[:2]
            scale_ratio = 640.0 / w_dim
            if scale_ratio < 1.0:
                disp_frame = cv2.resize(annotated, (640, int(h_dim * scale_ratio)))
            else:
                disp_frame = annotated
            _, buffer_img = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpg_as_text = base64.b64encode(buffer_img).decode('utf-8')

            if out_video is not None:
                out_video.write(annotated)

            elapsed_total = time.time() - start_time
            running_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0.0
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 100.0

            # Output frame details (with new SOTA fields)
            print(json.dumps({
                "type": "frame",
                "frame": f"data:image/jpeg;base64,{jpg_as_text}",
                "progress": round(progress, 1),
                "fps": round(running_fps, 1),
                "frame_idx": frame_idx,
                "alert_state": alert_state,
                "raw_prob": float(raw_prob),
                "calibrated_prob": float(confidence), # Map confidence to calibrated_prob for UI
                "accident_confidence": round(confidence * 100, 1),
                "risk_level": risk_level,
                "triggered_features": [k for k, v in features.items() if v],
                "feature_explanations": explanations,
                "vehicle_ids": list(vehicle_speeds.keys()),
                "vehicle_speeds": vehicle_speeds,
                "collision_pair": collision_pair,
                "features": {k: bool(v) if isinstance(v, (bool, np.bool_)) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in features.items()} if features else {}
            }), flush=True)

            if not is_live:
                elapsed = time.time() - loop_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # Release resources
    cap.release()
    if out_video is not None:
        out_video.release()

    print(json.dumps({
        "type": "complete",
        "output_video": output_filename,
        "total_frames": frame_idx
    }), flush=True)

if __name__ == "__main__":
    main()
