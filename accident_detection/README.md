# 🛡️ Aegis Eye — Real-time Autonomous Accident Detection System

Aegis Eye is a full-stack, production-grade Computer Vision and Deep Learning system designed to autonomously monitor traffic camera streams. It detects vehicle collisions, identifies helmet violations, and isolates threats in real time using a highly interactive React-based dashboard.

---

## 🏗️ Architecture & Technology Stack

The project has been separated into a modern web stack:

- **Frontend**: React.js, Vite, and CSS for a dynamic, sleek dashboard.
- **Backend Server**: Node.js and Express.js handling REST APIs and WebSocket streams.
- **Machine Learning Core**: Python, PyTorch, and YOLOv8 for running complex video processing, collision tracking, and inference.

---

## 🚀 Installation & Local Setup Guide

Follow these steps carefully to install and run the project on a new PC.

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Node.js**: v18+ (Download from [nodejs.org](https://nodejs.org/))
- **Python**: v3.9 - 3.11 (Download from [python.org](https://www.python.org/downloads/))
- **Git**: (Download from [git-scm.com](https://git-scm.com/))

### 2. Clone the Repository
Open a terminal (or Command Prompt) and run:
```bash
git clone <your-repository-url>
cd accident_detection
```

### 3. Setup the Python Machine Learning Environment
The Python scripts run the AI models behind the scenes. It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

### 4. Setup the Node.js Backend Server
The Node.js server coordinates the AI models and the React frontend. Open a **new terminal window/tab**, navigate to the `backend` folder, and install its dependencies.

```bash
cd backend
npm install
```

### 5. Setup the React Frontend
The frontend is the graphical dashboard you interact with. Open a **third terminal window/tab**, navigate to the `frontend` folder, and install its dependencies.

```bash
cd frontend
npm install
```

---

## 🏃‍♂️ How to Run the Project

To run the full system, you need to start **both** the backend and the frontend servers simultaneously.

### Step 1: Start the Backend Server
In your backend terminal, run:
```bash
cd backend
npm run dev
```
*(The server will start running, usually on `http://localhost:5000` or `http://localhost:3000`)*

### Step 2: Start the Frontend Dashboard
In your frontend terminal, run:
```bash
cd frontend
npm run dev
```
*(Vite will spin up the frontend, usually accessible at `http://localhost:5173`)*

### Step 3: Access the Dashboard
Open your web browser and go to the link provided by the frontend terminal (e.g., `http://localhost:5173`). 

---

## 📝 Usage

1. **Dashboard Controls**: Use the sidebar to switch between "Accident Detection" and "Helmet Detection".
2. **Video Input**: Upload an `.mp4` file or provide a stream URL.
3. **Live Monitoring**: The system will automatically spawn Python workers in the background, process the video stream frame-by-frame, and stream the results (including bounding boxes and alerts) directly to your React interface via WebSockets.

Enjoy exploring Aegis Eye!
