# AI-Powered Smart Traffic Management System

This project is an AI-powered smart traffic management system that uses computer vision (OpenCV) and deep learning (YOLOv8) to analyze real-time traffic footage, calculate vehicle density on a grid, and dynamically predict optimal traffic signal timings using a Machine Learning model.

## Core Features
1. **Vehicle Detection:** Utilizes YOLOv8n for real-time detection of cars, trucks, buses, and motorcycles.
2. **Grid-Based Analysis:** Divides the camera feed into a configurable grid (e.g., 3x3) to assess vehicle spread, calculate occupancy, and flag highly congested zones.
3. **Density Calculation:** Automatically categorizes traffic as LOW, MEDIUM, or HIGH based on vehicle counts and congestion scores.
4. **Machine Learning Signal Control:** A pre-trained Random Forest model predicts the optimal green light duration based on the live traffic stats.
5. **Modern Dashboard UI:** A clean, dark-mode overlay showing traffic stats, FPS, bounding boxes, grid heatmaps, and a live traffic light indicator.

## Project Structure
- `main.py`: Entry point for the application. Handles video input and runs the system loop.
- `detector.py`: YOLOv8 wrapper for detecting vehicles in the frame.
- `grid_analyzer.py`: Logic for mapping vehicles to a grid, determining occupancy and density.
- `ml_model.py`: Handles training the baseline Random Forest model and running predictions.
- `signal_controller.py`: Manages the traffic signal state machine (RED/GREEN/YELLOW logic).
- `utils.py`: Configuration constants and UI drawing functions.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Optional: CUDA-compatible GPU for accelerated YOLOv8 inference.

### 2. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

Note: If you have a CUDA-enabled GPU, you might want to install PyTorch with CUDA support. Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for the command specific to your system.

### 3. Running the System
You can run the system using a webcam or a pre-recorded traffic video.

**Using Webcam:**
```bash
python main.py
```
*(By default, `--source 0` is used)*

**Using a Video File:**
```bash
python main.py --source path/to/your/traffic_video.mp4
```

### 4. Quitting
Press the `q` key while the video window is focused to exit the application.

## How it Works
1. **Detection:** Frame is passed to YOLOv8 to find bounding boxes.
2. **Grid Mapping:** Vehicle centers are mapped to a 3x3 grid to evaluate localized congestion.
3. **ML Prediction:** Total vehicles and congestion score are sent to the ML model to generate a custom green light duration.
4. **Controller Update:** The signal controller counts down the current signal and transitions dynamically.
5. **UI Drawing:** All info is rendered in real-time on the output frame.
