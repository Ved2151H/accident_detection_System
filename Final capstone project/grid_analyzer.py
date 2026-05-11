import cv2
import utils

class GridAnalyzer:
    def __init__(self, rows=utils.GRID_ROWS, cols=utils.GRID_COLS):
        self.rows = rows
        self.cols = cols
        
    def analyze(self, frame, detections):
        """
        Divides the frame into a grid and calculates occupancy based on detections.
        Returns the annotated frame and grid statistics.
        """
        h, w = frame.shape[:2]
        # We might want to only analyze the left part, ignoring the dashboard area
        dash_width = 320
        active_w = w - dash_width
        
        cell_h = h // self.rows
        cell_w = active_w // self.cols
        
        grid_counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Calculate vehicle counts per grid
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Center of bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check if within active area
            if cx < active_w:
                col_idx = cx // cell_w
                row_idx = cy // cell_h
                
                # Make sure indices are within bounds
                col_idx = min(col_idx, self.cols - 1)
                row_idx = min(row_idx, self.rows - 1)
                
                grid_counts[row_idx][col_idx] += 1
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        total_vehicles = len(detections)
        occupied_grids = 0
        
        # Draw grids and calculate metrics
        for r in range(self.rows):
            for c in range(self.cols):
                gx1 = c * cell_w
                gy1 = r * cell_h
                gx2 = gx1 + cell_w
                gy2 = gy1 + cell_h
                
                count = grid_counts[r][c]
                
                color = utils.COLOR_GRID
                thickness = 1
                
                if count > 0:
                    occupied_grids += 1
                    if count >= 3:
                        # Heavily occupied grid
                        color = utils.COLOR_GRID_HEAVY
                        thickness = 2
                        
                        # Add light red overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                        
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, thickness)
                cv2.putText(frame, f"N:{count}", (gx1 + 10, gy1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Calculate congestion score (0 to 1)
        # Based on occupied grids and total vehicles
        max_capacity_est = self.rows * self.cols * 3 # Assuming 3 cars per grid is heavy
        congestion_score = min(total_vehicles / max_capacity_est, 1.0)
        
        # Determine Density
        if total_vehicles <= utils.DENSITY_LOW_THRESHOLD:
            density = "LOW"
        elif total_vehicles <= utils.DENSITY_MEDIUM_THRESHOLD:
            density = "MEDIUM"
        else:
            density = "HIGH"
            
        stats = {
            "total_vehicles": total_vehicles,
            "occupied_grids": occupied_grids,
            "congestion_score": congestion_score,
            "density": density,
            "grid_counts": grid_counts
        }
        
        return frame, stats
