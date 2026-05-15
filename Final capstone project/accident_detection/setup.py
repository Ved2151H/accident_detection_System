"""
Day 1 Setup Script — Accident Detection System
Run: python setup.py
"""

import os
import subprocess
import sys

FOLDERS = [
    "data/raw",
    "data/processed",
    "data/processed/frames",
    "data/processed/labels",
    "models",
    "pipeline",
    "alerts",
    "dashboard",
    "tests",
    "logs",
]

def create_structure():
    print("\n[1/4] Creating project folder structure...")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        # add .gitkeep so empty folders are tracked
        gitkeep = os.path.join(folder, ".gitkeep")
        if not os.listdir(folder):
            open(gitkeep, "w").close()
    print("      Done.")

def create_env_file():
    print("[2/4] Creating .env template...")
    env_content = """# --- Alert Config ---
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_NUMBER=+1XXXXXXXXXX
ALERT_TO_NUMBER=+1XXXXXXXXXX

# --- Email Alert Config ---
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password_here
ALERT_TO_EMAIL=recipient@gmail.com

# --- Model Config ---
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
FRAME_SKIP=5
"""
    with open(".env", "w") as f:
        f.write(env_content)
    print("      .env created — fill in your credentials before Day 4.")

def create_gitignore():
    print("[3/4] Creating .gitignore...")
    gitignore = """.env
__pycache__/
*.pyc
*.pt
*.pth
data/raw/
data/processed/
logs/
*.egg-info/
dist/
.DS_Store
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    print("      Done.")

def install_requirements():
    print("[4/4] Installing requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False
    )
    if result.returncode == 0:
        print("      All packages installed successfully.")
    else:
        print("      Some packages failed. Check errors above.")

def verify_imports():
    print("\n--- Verifying key imports ---")
    checks = [
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv8 (Ultralytics)"),
        ("streamlit", "Streamlit"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]
    all_ok = True
    for module, name in checks:
        try:
            __import__(module)
            print(f"  [OK]  {name}")
        except ImportError:
            print(f"  [FAIL] {name} — run: pip install {module}")
            all_ok = False

    if all_ok:
        print("\nAll imports OK. You are ready for Day 2!")
    else:
        print("\nSome imports failed. Fix them before proceeding.")

if __name__ == "__main__":
    print("=" * 50)
    print("  Accident Detection System — Day 1 Setup")
    print("=" * 50)
    create_structure()
    create_env_file()
    create_gitignore()
    install_requirements()
    verify_imports()
    print("\nSetup complete. Check PLAN.md next.")