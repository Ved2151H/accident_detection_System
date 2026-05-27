import argparse
import sys
import os
from pathlib import Path

# Ensure UTF-8 console output encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add workspace directory to python path
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Accident Detection System - Retraining Dashboard Control")
    parser.add_argument(
        "--mode", 
        choices=["yolo", "lstm", "all"], 
        default="all",
        help="Specify training mode: 'yolo' to fine-tune YOLOv8 classifier, 'lstm' to train LSTM Temporal Sequence model, or 'all' to run both sequentially."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default=None, help="Device to train on (e.g. 0, cpu).")

    args = parser.parse_args()

    print("=" * 60)
    print("SYSTEM - ACCIDENT DETECTION RETRAINING SUITE")
    print("=" * 60)
    print(f"Target Mode: {args.mode.upper()}")
    if args.epochs:
        print(f"Custom Epochs: {args.epochs}")
    if args.batch:
        print(f"Custom Batch Size: {args.batch}")
    if args.device:
        print(f"Custom Device: {args.device}")
    print("-" * 60)

    if args.mode in ["yolo", "all"]:
        print("\n[PHASE 1] Fine-tuning YOLOv8 Classification Model...")
        import training.train_yolo as ty
        
        # Override config parameters if provided
        if args.epochs:
            ty.EPOCHS = args.epochs
        if args.batch:
            ty.BATCH_SIZE = args.batch
        if args.device:
            ty.DEVICE = int(args.device) if args.device.isdigit() else args.device

        ty.convert_to_yolo_format()
        ty.train_yolo_classifier()
        ty.evaluate_model()
        print("[OK] YOLOv8 Classification Model Training Complete.")

    if args.mode in ["lstm", "all"]:
        print("\n[PHASE 2] Training LSTM Temporal Anomaly Model...")
        import training.train_lstm as tl
        
        # Override config parameters if provided
        if args.epochs:
            tl.EPOCHS = args.epochs
        if args.batch:
            tl.BATCH_SIZE = args.batch
        if args.device:
            import torch
            tl.DEVICE = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

        tl.train()
        print("[OK] LSTM Temporal Anomaly Model Training Complete.")

    print("\n" + "=" * 60)
    print("[SUCCESS] MODEL TRAINING COMPLETE AND WEIGHTS UPDATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

