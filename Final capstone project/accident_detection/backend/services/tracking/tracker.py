"""
ByteTrack-based Rider and Helmet Association Tracker.
Handles association between motorcycles, riders, and helmets,
and maintains robust statistics using unique track IDs.
"""

import numpy as np

def get_box_overlap(box_a, box_b):
    """
    Computes Intersection over Area (IoA) relative to box_a,
    and Intersection over Union (IoU) between box_a and box_b.
    Box format: [x1, y1, x2, y2]
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Intersection coordinates
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0, 0.0

    area_a = max(1.0, (x2_a - x1_a) * (y2_a - y1_a))
    area_b = max(1.0, (x2_b - x1_b) * (y2_b - y1_b))

    union_area = area_a + area_b - inter_area
    iou = inter_area / float(union_area)
    ioa = inter_area / float(area_a)  # Overlap relative to box_a

    return iou, ioa


class HelmetTracker:
    def __init__(self, history_len=15):
        self.history_len = history_len
        # Track registry: maps motorcycle track ID (int) -> dict
        # {
        #    "rider_id": int or None,
        #    "helmet_history": list of bool,
        #    "last_bike_box": list of float,
        #    "last_rider_box": list of float,
        #    "compliance_logged": bool,
        #    "snapshot_saved": bool
        # }
        self.tracks = {}
        
        # Unique tracking sets
        self.unique_bikes = set()
        self.helmet_users = set()
        self.non_helmet_riders = set()

    def reset(self):
        """Resets the tracking state."""
        self.tracks.clear()
        self.unique_bikes.clear()
        self.helmet_users.clear()
        self.non_helmet_riders.clear()

    def update(self, motorcycles, riders, helmet_detections):
        """
        Updates the tracker with detections from the current frame.
        
        Parameters:
        - motorcycles: list of dict, each: {"box": [x1,y1,x2,y2], "id": int, "conf": float}
        - riders: list of dict, each: {"box": [x1,y1,x2,y2], "id": int, "conf": float}
        - helmet_detections: list of dict, each: {"box": [x1,y1,x2,y2], "class": str, "conf": float}
        
        Returns:
        - active_associations: list of dict, containing details of current matches for drawing
        """
        active_associations = []
        current_bike_ids = set()

        # Step 1: Process each motorcycle in this frame
        for bike in motorcycles:
            bike_id = bike["id"]
            bike_box = bike["box"]
            
            if bike_id is None:
                continue
                
            self.unique_bikes.add(bike_id)
            current_bike_ids.add(bike_id)

            # Initialize track if new
            if bike_id not in self.tracks:
                self.tracks[bike_id] = {
                    "rider_id": None,
                    "helmet_history": [],
                    "last_bike_box": bike_box,
                    "last_rider_box": None,
                    "compliance_logged": False,
                    "snapshot_saved": False
                }
            
            track = self.tracks[bike_id]
            track["last_bike_box"] = bike_box

            # Step 2: Associate rider (person) with this bike using box overlap (IoA of rider inside bike)
            best_rider_id = None
            best_rider_box = None
            max_overlap = 0.0

            for rider in riders:
                rider_box = rider["box"]
                rider_id = rider["id"]
                
                # Calculate how much the rider overlaps with the bike
                # A rider sits "on/in" the bike, so we check intersection / area(rider)
                _, ioa_rider = get_box_overlap(rider_box, bike_box)
                
                if ioa_rider > 0.30 and ioa_rider > max_overlap:
                    max_overlap = ioa_rider
                    best_rider_id = rider_id
                    best_rider_box = rider_box

            if best_rider_id is not None:
                track["rider_id"] = best_rider_id
                track["last_rider_box"] = best_rider_box

                # Step 3: Check helmet/no_helmet overlap with rider's box
                has_helmet = False
                has_no_helmet = False
                best_helmet_conf = 0.0
                best_no_helmet_conf = 0.0
                matched_helmet_box = None

                for h_det in helmet_detections:
                    h_box = h_det["box"]
                    h_cls = h_det["class"]
                    h_conf = h_det["conf"]

                    # Check how much of the helmet box lies inside the rider's box
                    _, ioa_h = get_box_overlap(h_box, best_rider_box)
                    if ioa_h > 0.40:
                        if h_cls == "helmet":
                            has_helmet = True
                            best_helmet_conf = max(best_helmet_conf, h_conf)
                            matched_helmet_box = h_box
                        elif h_cls == "no_helmet":
                            has_no_helmet = True
                            best_no_helmet_conf = max(best_no_helmet_conf, h_conf)
                            matched_helmet_box = h_box

                # Decide this frame's vote
                if has_helmet:
                    vote = True
                    conf_score = best_helmet_conf
                elif has_no_helmet:
                    vote = False
                    conf_score = best_no_helmet_conf
                else:
                    # Default to no_helmet if rider is present but nothing is classified
                    vote = False
                    conf_score = 0.0
                    # Estimate a head box at the top of the rider box for visualization if no helmet box
                    rw = best_rider_box[2] - best_rider_box[0]
                    rh = best_rider_box[3] - best_rider_box[1]
                    matched_helmet_box = [
                        best_rider_box[0] + rw * 0.15,
                        best_rider_box[1],
                        best_rider_box[2] - rw * 0.15,
                        best_rider_box[1] + rh * 0.25
                    ]

                # Update history
                history = track["helmet_history"]
                history.append(vote)
                if len(history) > self.history_len:
                    history.pop(0)

                # Determine rolling compliance status (requires at least 3 votes for stability)
                if len(history) >= 3:
                    is_compliant = sum(history) > (len(history) / 2)
                else:
                    is_compliant = vote  # Fallback to current frame vote

                # Update global unique compliance sets
                if is_compliant:
                    self.helmet_users.add(bike_id)
                    self.non_helmet_riders.discard(bike_id)
                else:
                    self.non_helmet_riders.add(bike_id)
                    self.helmet_users.discard(bike_id)

                # Add to active associations for the frame render
                active_associations.append({
                    "bike_id": bike_id,
                    "bike_box": bike_box,
                    "rider_id": best_rider_id,
                    "rider_box": best_rider_box,
                    "head_box": matched_helmet_box,
                    "helmet_present": is_compliant,
                    "confidence": conf_score,
                    "compliance_logged": track["compliance_logged"],
                    "snapshot_saved": track["snapshot_saved"]
                })

        # Remove dead tracks that haven't been seen in the current frame to keep memory clean
        # (Only keep unique stats, but clean active state)
        for dead_id in list(self.tracks.keys()):
            if dead_id not in current_bike_ids:
                # Keep the track configuration for unique counts, but clear historical list if too old
                pass

        return active_associations

    def get_stats(self):
        """Returns the accumulated tracking statistics."""
        total_bikes = len(self.unique_bikes)
        helmet_users = len(self.helmet_users)
        non_helmet = len(self.non_helmet_riders)
        total_riders = helmet_users + non_helmet
        compliance = (helmet_users / max(1, total_riders)) * 100.0 if total_riders > 0 else 100.0

        return {
            "total_bikes": total_bikes,
            "helmet_users": helmet_users,
            "non_helmet_riders": non_helmet,
            "compliance_pct": compliance
        }

    def mark_logged(self, bike_id):
        """Marks a track as logged and snapshot saved to prevent duplicates."""
        if bike_id in self.tracks:
            self.tracks[bike_id]["compliance_logged"] = True
            self.tracks[bike_id]["snapshot_saved"] = True

