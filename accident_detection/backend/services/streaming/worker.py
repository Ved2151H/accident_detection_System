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
    parser.add_argument("--task", type=str, required=True, choices=["collision", "helmet"], help="Task to run")
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

    # Encode first frame to base64
    _, buffer_img = cv2.imencode('.jpg', first_frame)
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
            _, buffer_img = cv2.imencode('.jpg', annotated)
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

    elif args.task == "helmet":
        import threading
        from backend.services.tracking.tracker import HelmetTracker
        
        model_container = {"base": None, "helmet": None, "ready": False, "error": None}
        
        def load_helmet_models_async():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(0.625, 0)
                from ultralytics import YOLO
                from backend.services.helmet.helmet_utils import download_helmet_model
                
                base_model = YOLO("models/helmet_detection/weights/yolov8n.pt")
                helmet_model_path = download_helmet_model("models/helmet_detection/weights/helmet_model.pt")
                helmet_model = YOLO(helmet_model_path)
                
                model_container["base"] = base_model
                model_container["helmet"] = helmet_model
                model_container["ready"] = True
            except Exception as e:
                model_container["error"] = str(e)
                
        load_thread = threading.Thread(target=load_helmet_models_async)
        load_thread.start()
        
        tracker = HelmetTracker()

        motorcycle_conf = 0.25
        rider_conf = 0.25
        helmet_conf = 0.25
        
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1

            motorcycles = []
            active_assoc = []
            stats = tracker.get_stats()
            annotated = frame.copy()

            if model_container["ready"]:
                base_model = model_container["base"]
                helmet_model = model_container["helmet"]
                
                # Base model tracking (classes=[0, 1, 3] person, bicycle, motorcycle)
                base_results = base_model.track(
                    frame, 
                    persist=True, 
                    tracker="bytetrack.yaml", 
                    classes=[0, 1, 3], 
                    conf=min(motorcycle_conf, rider_conf),
                    verbose=False,
                    imgsz=480
                )
                
                riders = []
                
                if base_results and len(base_results) > 0:
                    boxes = base_results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            xyxy = box.xyxy[0].tolist()
                            track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else None
                            
                            if (cls_id == 1 or cls_id == 3) and conf >= motorcycle_conf:
                                motorcycles.append({"box": xyxy, "id": track_id, "conf": conf})
                            elif cls_id == 0 and conf >= rider_conf:
                                riders.append({"box": xyxy, "id": track_id, "conf": conf})
    
                # Helmet detection
                helmet_results = helmet_model(frame, conf=helmet_conf, verbose=False, imgsz=480)
                helmet_detections = []
                
                if helmet_results and len(helmet_results) > 0:
                    h_boxes = helmet_results[0].boxes
                    if h_boxes is not None:
                        for box in h_boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            xyxy = box.xyxy[0].tolist()
                            raw_name = helmet_model.names[cls_id].lower()
                            h_cls = "unknown"
                            if "no_helmet" in raw_name or "no-helmet" in raw_name or "without" in raw_name or "head" in raw_name:
                                h_cls = "no_helmet"
                            elif "helmet" in raw_name or "with_helmet" in raw_name:
                                h_cls = "helmet"
                            helmet_detections.append({"box": xyxy, "class": h_cls, "conf": conf})
    
                active_assoc = tracker.update(motorcycles, riders, helmet_detections)
                stats = tracker.get_stats()
    
                # Annotate frame
                for assoc in active_assoc:
                    bike_box = assoc["bike_box"]
                    head_box = assoc["head_box"]
                    helmet_present = assoc["helmet_present"]
                    bike_id = assoc["bike_id"]
                    conf_score = assoc["confidence"]
                    
                    bx1, by1, bx2, by2 = map(int, bike_box)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"Bike #{bike_id}", (bx1, max(by1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    if head_box is not None:
                        hx1, hy1, hx2, hy2 = map(int, head_box)
                        box_color = (255, 0, 0) if helmet_present else (0, 0, 255)
                        label_str = f"Helmet OK {conf_score:.0%}" if helmet_present else "No Helmet!"
                        
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), box_color, 2)
                        cv2.putText(annotated, label_str, (hx1, max(hy1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                        
                        # Handle violation logging
                        if not helmet_present and not assoc["compliance_logged"]:
                            from backend.services.helmet.helmet_utils import save_evidence_image, log_violation_to_csv
                            snap_frame = annotated.copy()
                            cv2.rectangle(snap_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                            snap_path = save_evidence_image(snap_frame, bike_id)
                            log_violation_to_csv(bike_id, "NO_HELMET_VIOLATION", snap_path)
                            tracker.mark_logged(bike_id)
                            
                            # Print violation notification to server.js
                            print(json.dumps({
                                "type": "violation",
                                "track_id": bike_id,
                                "status": "No Helmet",
                                "snapshot": snap_path,
                                "timestamp": datetime.datetime.now().isoformat()
                            }), flush=True)
            else:
                if model_container["error"]:
                    print(json.dumps({"type": "error", "message": f"AI init failed: {model_container['error']}"}), flush=True)
                    sys.exit(1)
                    
                # Display raw frame with initializing watermark
                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 50), (0, 10, 20), -1)
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                
                dots = "." * (int(time.time() * 2) % 4)
                text = f"AEGIS EYE: Initializing Helmet Scan AI{dots}"
                cv2.putText(annotated, text, (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

            # Draw live FPS
            elapsed_total = time.time() - start_time
            running_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0.0
            cv2.putText(annotated, f"FPS: {running_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Write to output file
            if out_video is not None:
                out_video.write(annotated)

            # Encode frame to base64
            _, buffer_img = cv2.imencode('.jpg', annotated)
            jpg_as_text = base64.b64encode(buffer_img).decode('utf-8')

            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 100.0

            # Indicators state
            bike_present = len(motorcycles) > 0
            rider_present = any(a["rider_id"] is not None for a in active_assoc)
            helmet_status = False
            if rider_present:
                helmet_status = all(a["helmet_present"] for a in active_assoc if a["rider_id"] is not None)

            # Output frame details
            print(json.dumps({
                "type": "frame",
                "frame": f"data:image/jpeg;base64,{jpg_as_text}",
                "progress": round(progress, 1),
                "fps": round(running_fps, 1),
                "frame_idx": frame_idx,
                "bike_present": bike_present,
                "rider_present": rider_present,
                "helmet_status": helmet_status,
                "stats": stats
            }), flush=True)

            if is_live:
                time.sleep(0.002)
            else:
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / fps) - elapsed
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
