import cv2
from ultralytics import YOLO
import utils

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 model for vehicle detection.
        Uses yolov8n.pt by default for real-time performance.
        """
        self.model = YOLO(model_path)
        
        # COCO class IDs for vehicles
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        
        # If using a custom trained model for emergency vehicles, map those IDs here
        # E.g., self.emergency_classes = [80] (ambulance), [81] (fire_truck)
        self.emergency_classes = [] 

        
    def detect(self, frame):
        """
        Runs object detection on the given frame.
        Returns the processed frame with bounding boxes and a list of detections.
        """
        results = self.model(frame, classes=self.vehicle_classes + self.emergency_classes, conf=utils.CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        has_emergency = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'class_name': cls_name,
                    'class_id': cls_id
                })
                
                # Draw bounding box and label
                color = (255, 100, 100)
                if cls_id in self.emergency_classes:
                    has_emergency = True
                    color = (0, 0, 255) # Red for emergency
                elif cls_name == "truck" or cls_name == "bus":
                    color = (100, 100, 255) # Reddish for large vehicles
                elif cls_name == "motorcycle":
                    color = (100, 255, 100) # Greenish for bikes
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        return frame, detections, has_emergency
