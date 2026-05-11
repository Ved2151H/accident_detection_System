import time

class SignalController:
    def __init__(self, ml_model):
        self.state = "RED"  # Initial state
        self.timer = 5      # Initial countdown in seconds
        self.last_update_time = time.time()
        
        self.ml_model = ml_model
        
        # State timings
        self.yellow_duration = 3
        self.min_red_duration = 5
        
    def update(self, stats):
        """
        Updates the signal state machine based on the time elapsed.
        Dynamically changes timings based on traffic stats when switching to GREEN.
        """
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Handle Emergency Priority
        if stats.get('has_emergency', False) and self.state != "GREEN":
            # Force immediate transition to green
            self.state = "GREEN"
            self.timer = 15 # Give at least 15 seconds for the emergency vehicle
            self.last_update_time = current_time
            stats['ml_prediction'] = "EMERGENCY OVERRIDE"
            return self.state, self.timer
            
        if elapsed >= 1.0:
            self.timer -= 1
            self.last_update_time = current_time
            
        if self.timer <= 0:
            self.transition_state(stats)
            
        return self.state, self.timer
        
    def transition_state(self, stats):
        """
        Handles state transitions and determines new countdown timer.
        RED -> GREEN -> YELLOW -> RED
        """
        if self.state == "RED":
            self.state = "GREEN"
            # Get optimal green time from ML model
            optimal_green = self.ml_model.predict_green_time(stats)
            stats['ml_prediction'] = optimal_green # Store for UI
            self.timer = optimal_green
            
        elif self.state == "GREEN":
            self.state = "YELLOW"
            self.timer = self.yellow_duration
            
        elif self.state == "YELLOW":
            self.state = "RED"
            # In a real system, the RED duration depends on the cross-traffic
            # Here we just use a default minimum red duration for demonstration
            self.timer = self.min_red_duration
