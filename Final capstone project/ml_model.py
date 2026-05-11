import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

class TrafficMLModel:
    def __init__(self, model_path="traffic_model.pkl"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_baseline_model()
            
    def train_baseline_model(self):
        """
        Trains a baseline Random Forest model using synthetic data.
        In a real scenario, this would use historical traffic data.
        """
        print("Training baseline ML model...")
        
        # Generate synthetic data
        # Features: [total_vehicles, occupied_grids, congestion_score]
        # Target: green_signal_duration (seconds)
        
        np.random.seed(42)
        n_samples = 1000
        
        total_vehicles = np.random.randint(0, 50, n_samples)
        occupied_grids = np.random.randint(0, 10, n_samples) # Assuming 3x3 grid (max 9)
        congestion_score = total_vehicles / 50.0 + np.random.normal(0, 0.05, n_samples)
        congestion_score = np.clip(congestion_score, 0, 1)
        
        # Green duration formula (synthetic ground truth)
        # Base 10s + 1.5s per vehicle + extra for high congestion
        green_duration = 10 + (total_vehicles * 1.5) + (congestion_score * 15)
        green_duration = np.clip(green_duration, 15, 60) # Min 15s, Max 60s
        
        X = pd.DataFrame({
            'total_vehicles': total_vehicles,
            'occupied_grids': occupied_grids,
            'congestion_score': congestion_score
        })
        
        y = green_duration
        
        self.model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        print("Model trained and saved to", self.model_path)
        
    def load_model(self):
        """Loads the pre-trained model from disk."""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("ML Model loaded successfully.")
        
    def predict_green_time(self, stats):
        """
        Predicts optimal green signal time based on current traffic stats.
        """
        if not self.model:
            return 15 # Default safe value
            
        features = pd.DataFrame({
            'total_vehicles': [stats.get('total_vehicles', 0)],
            'occupied_grids': [stats.get('occupied_grids', 0)],
            'congestion_score': [stats.get('congestion_score', 0.0)]
        })
        
        pred = self.model.predict(features)[0]
        
        # Return as integer seconds, capped between 10 and 60
        return int(np.clip(pred, 10, 60))
