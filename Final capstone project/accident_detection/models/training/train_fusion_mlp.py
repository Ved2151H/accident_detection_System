import os
import sys
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.config import DEVICE, SEQUENCE_LEN
from dashboard.models import load_models
from dashboard.telemetry import run_inference, AccidentFusionMLP

def train_mlp():
    print("[1/5] Initializing models for feature extraction...")
    yolo_model, lstm_model = load_models()
    if yolo_model is None or lstm_model is None:
        print("[-] Error: YOLO or LSTM model weights not found! Train them first.")
        sys.exit(1)
        
    print("[2/5] Scanning preprocessed splits for training data...")
    train_dir = Path("data/processed/frames/train")
    
    # Group frames by video stem
    video_sequences = {}
    
    for cls in ["normal", "accident"]:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            continue
        for fpath in cls_dir.glob("*.jpg"):
            stem = fpath.stem
            parts = stem.split("_")
            # video_id is everything between class name and frame index
            video_id = "_".join(parts[1:-1])
            frame_idx = int(parts[-1])
            
            if video_id not in video_sequences:
                video_sequences[video_id] = []
            video_sequences[video_id].append((frame_idx, cls, fpath))
            
    print(f"  Found {len(video_sequences)} video sequences in train split.")
    
    X_list = []
    y_list = []
    
    print("[3/5] Extracting SOTA 24-dimensional feature vectors...")
    # Limit number of sequences to keep training under 2 minutes
    limit_vids = list(video_sequences.keys())[:100]
    
    for vid_idx, vid_id in enumerate(limit_vids):
        seq_frames = sorted(video_sequences[vid_id], key=lambda x: x[0])
        
        # Initialize temporal trackers
        buffer = deque(maxlen=SEQUENCE_LEN)
        vehicle_tracks = {}
        last_small_gray_container = [None]
        
        if vid_idx % 10 == 0:
            print(f"  Processing sequence {vid_idx}/{len(limit_vids)}...")
            
        for frame_idx, frame_cls, fpath in seq_frames:
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue
                
            # Run inference to get the raw features
            # Pass dummy state machine and gray frame container
            _, _, _, _, meta_info = run_inference(
                frame, buffer, yolo_model, lstm_model, vehicle_tracks, 
                state_machine=None, last_small_gray_container=last_small_gray_container
            )
            
            raw_features = meta_info.get("raw_features")
            if raw_features is not None:
                X_list.append(raw_features)
                y_list.append(1 if frame_cls == "accident" else 0)
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"  Extracted {len(X)} feature vectors.")
    print(f"  Accident class count: {np.sum(y == 1)}, Normal class count: {np.sum(y == 0)}")
    
    # Compute mean and standard deviation
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Save scaler configuration
    scaler_config = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/fusion_scaler.json", "w") as f:
        json.dump(scaler_config, f, indent=2)
    print("[+] Saved models/fusion_scaler.json")
    
    # Standardize data
    X_normalized = (X - mean) / (std + 1e-6)
    
    # Build PyTorch Dataset
    dataset = TensorDataset(torch.from_numpy(X_normalized), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("[4/5] Training PyTorch MLP Fusion Classifier...")
    model = AccidentFusionMLP(input_dim=24, hidden_dim=64, output_dim=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 80
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        avg_loss = epoch_loss / total if total > 0 else 0.0
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")
            
    # Save model weights
    torch.save(model.state_dict(), "models/fusion_mlp.pt")
    print("[+] Saved models/fusion_mlp.pt")
    print("[5/5] MLP Fusion Classifier training complete!")

if __name__ == "__main__":
    train_mlp()
