import cv2
import time
import argparse
import sys
from detector import VehicleDetector
from grid_analyzer import GridAnalyzer
from ml_model import TrafficMLModel
from signal_controller import SignalController
import utils

def main():
    parser = argparse.ArgumentParser(description="AI Smart Traffic Management System")
    parser.add_argument('--source', type=str, default='0', help='Video source: webcam index (e.g. 0) or path to video file')
    args = parser.parse_args()

    # Determine video source
    source = args.source
    if source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    # Initialize components
    print("Initializing components...")
    detector = VehicleDetector()
    analyzer = GridAnalyzer()
    ml_model = TrafficMLModel()
    controller = SignalController(ml_model)

    # FPS Calculation variables
    prev_time = 0

    print("System started successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            # If it's a video file, loop it
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Resize frame for consistent processing and display
        frame = cv2.resize(frame, (1280, 720))

        # 1. Vehicle Detection
        frame, detections, has_emergency = detector.detect(frame)

        # 2. Grid-based Analysis
        frame, stats = analyzer.analyze(frame, detections)
        stats['has_emergency'] = has_emergency

        # 3. Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        stats['fps'] = fps

        # 4. Signal Controller Update
        signal_state, countdown = controller.update(stats)

        # 5. Dashboard / Visualization
        frame = utils.draw_dashboard(frame, stats, signal_state, countdown)

        # Display output
        cv2.imshow("Smart Traffic Management", frame)

        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting system...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
