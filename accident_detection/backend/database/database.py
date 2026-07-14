import sqlite3
import os
import datetime
import pandas as pd
from pathlib import Path
import cv2
from backend.utils.config import DB_PATH, SNAPSHOTS_DIR

def init_db():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, source TEXT,
        yolo_conf REAL, lstm_prob REAL, snapshot TEXT,
        latitude REAL, longitude REAL, digipin TEXT)""")
    
    # Database migration checks
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(incidents)")
        columns = [row[1] for row in cursor.fetchall()]
        if "latitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN latitude REAL")
        if "longitude" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN longitude REAL")
        if "digipin" not in columns:
            cursor.execute("ALTER TABLE incidents ADD COLUMN digipin TEXT")
    except Exception as e:
        print(f"DB Migration Error: {e}")
    conn.commit()
    conn.close()

def log_incident(source, yolo_conf, lstm_prob, snapshot_path, latitude, longitude, digipin):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO incidents (timestamp, source, yolo_conf, lstm_prob, snapshot, latitude, longitude, digipin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), source, yolo_conf, lstm_prob, snapshot_path, latitude, longitude, digipin))
    conn.commit()
    conn.close()

def get_incidents():
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", conn)
    conn.close()
    return df

def save_snapshot(frame):
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOTS_DIR, f"incident_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
