import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize

# Set VRAM limit: 5GB of 8GB is 0.625
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.625, 0)
    print("[v] VRAM limit set to 5GB (0.625 fraction)")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_lstm import AccidentLSTM, SequenceDataset, val_transform, DEVICE, BATCH_SIZE

def calibrate():
    print("=" * 55)
    print("  Accident Detection System — Temperature Calibration")
    print("=" * 55)
    
    # 1. Load validation dataset
    print("[1/3] Loading validation dataset...")
    val_ds = SequenceDataset("val", val_transform)
    if len(val_ds) == 0:
        print("[ERROR] No validation sequences found. Run preprocessing and training first.")
        return
        
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Load trained LSTM model
    print("[2/3] Loading trained LSTM model weights...")
    model_path = os.path.join("models", "lstm_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}. Train the LSTM model first.")
        return
        
    model = AccidentLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Collect logits and labels
    print("  Collecting model predictions on validation set...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, labels in val_dl:
            seqs = seqs.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                logits = model(seqs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 3. Optimize temperature scaling
    print("[3/3] Optimizing temperature parameter T...")
    
    def loss_func(T):
        t_val = T[0]
        # Avoid division by zero or negative T
        if t_val <= 0.01:
            return 1e9
        scaled_logits = all_logits / t_val
        loss = nn.CrossEntropyLoss()(scaled_logits, all_labels)
        return loss.item()
        
    res = minimize(loss_func, x0=[1.0], method='Nelder-Mead', bounds=[(0.1, 10.0)])
    best_T = float(res.x[0])
    
    initial_loss = loss_func([1.0])
    calibrated_loss = loss_func([best_T])
    
    print(f"\n  Optimal Temperature  : {best_T:.6f}")
    print(f"  Initial Loss (T=1.0) : {initial_loss:.6f}")
    print(f"  Calibrated Loss      : {calibrated_loss:.6f}")
    
    # Save parameters
    calibration_data = {
        "temperature": best_T,
        "initial_bce_loss": initial_loss,  # naming bce_loss for fallback compatibility
        "calibrated_bce_loss": calibrated_loss
    }
    
    os.makedirs("models", exist_ok=True)
    cal_path = os.path.join("models", "calibration.json")
    with open(cal_path, "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"\n[SUCCESS] Calibration saved to: {cal_path}")
    print("=" * 55)

if __name__ == "__main__":
    calibrate()
