import cv2
import numpy as np

# Configuration Settings
GRID_ROWS = 3
GRID_COLS = 3
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Traffic Density Thresholds
DENSITY_LOW_THRESHOLD = 5
DENSITY_MEDIUM_THRESHOLD = 15

# UI Colors (BGR)
COLOR_BG_DARK = (25, 25, 25)
COLOR_TEXT = (240, 240, 240)
COLOR_ACCENT = (255, 150, 0) # Cyan-like or Orange
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_GRID = (100, 100, 100)
COLOR_GRID_HEAVY = (0, 0, 200)

def draw_dashboard(frame, stats, signal_state, countdown):
    """
    Draws a dark modern interface dashboard on the right side or top of the frame.
    We will overlay a dark semi-transparent rectangle on the right side.
    """
    h, w = frame.shape[:2]
    
    # Dashboard layout parameters
    dash_width = 320
    dash_x = w - dash_width
    
    # Draw dark background for dashboard
    overlay = frame.copy()
    cv2.rectangle(overlay, (dash_x, 0), (w, h), COLOR_BG_DARK, -1)
    
    # Alpha blending for semi-transparent dashboard
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw Title
    cv2.putText(frame, "SMART TRAFFIC", (dash_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2)
    cv2.putText(frame, "MANAGEMENT", (dash_x + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2)
    cv2.line(frame, (dash_x + 20, 85), (w - 20, 85), (100, 100, 100), 1)

    # Signal Indicator
    cv2.putText(frame, "CURRENT SIGNAL:", (dash_x + 20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    # Draw Traffic Light
    light_y = 180
    cv2.rectangle(frame, (dash_x + 100, light_y - 20), (dash_x + 160, light_y + 140), (40, 40, 40), -1)
    cv2.rectangle(frame, (dash_x + 100, light_y - 20), (dash_x + 160, light_y + 140), (100, 100, 100), 2)
    
    # Base colors
    r_color = (0, 0, 80)
    y_color = (0, 80, 80)
    g_color = (0, 80, 0)
    
    if signal_state == "RED":
        r_color = COLOR_RED
    elif signal_state == "YELLOW":
        y_color = COLOR_YELLOW
    elif signal_state == "GREEN":
        g_color = COLOR_GREEN
        
    cv2.circle(frame, (dash_x + 130, light_y + 10), 20, r_color, -1)
    cv2.circle(frame, (dash_x + 130, light_y + 60), 20, y_color, -1)
    cv2.circle(frame, (dash_x + 130, light_y + 110), 20, g_color, -1)
    
    # Countdown timer
    cv2.putText(frame, f"{countdown:02d}s", (dash_x + 180, light_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 3)
    
    cv2.line(frame, (dash_x + 20, light_y + 160), (w - 20, light_y + 160), (100, 100, 100), 1)

    # Statistics Panel
    stats_y = light_y + 200
    cv2.putText(frame, "TRAFFIC STATS", (dash_x + 20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
    
    stats_lines = [
        f"Total Vehicles: {stats.get('total_vehicles', 0)}",
        f"Density Level: {stats.get('density', 'UNKNOWN')}",
        f"Congestion Score: {stats.get('congestion_score', 0.0):.2f}",
        f"Occupied Grids: {stats.get('occupied_grids', 0)} / {GRID_ROWS * GRID_COLS}",
        f"FPS: {stats.get('fps', 0):.1f}"
    ]
    
    y_offset = stats_y + 30
    for line in stats_lines:
        cv2.putText(frame, line, (dash_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
        y_offset += 30

    # Draw ML Prediction Info if available
    if 'ml_prediction' in stats:
        cv2.line(frame, (dash_x + 20, y_offset), (w - 20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        cv2.putText(frame, "AI PREDICTION:", (dash_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 1)
        y_offset += 25
        pred_text = str(stats['ml_prediction'])
        if isinstance(stats['ml_prediction'], int):
            pred_text = f"{pred_text}s"
        cv2.putText(frame, f"Suggested Green: {pred_text}", (dash_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
        
    # Draw Emergency Alert
    if stats.get('has_emergency', False):
        y_offset += 30
        cv2.rectangle(frame, (dash_x + 20, y_offset - 20), (w - 20, y_offset + 10), (0, 0, 200), -1)
        cv2.putText(frame, "EMERGENCY DETECTED", (dash_x + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame
