import numpy as np
import streamlit as st
from backend.utils.digipin_helper import DigiPinHelper

digipin_helper = DigiPinHelper()

# ── Realistic GPS Coordinate Generator ───────────────────
def generate_gps():
    # Randomly selects a major region across different corners of India for video testing
    cities = [
        {"name": "New Delhi (North)", "lat": 28.6139, "lon": 77.2090},
        {"name": "Mumbai (West)", "lat": 19.0760, "lon": 72.8777},
        {"name": "Kolkata (East)", "lat": 22.5726, "lon": 88.3639},
        {"name": "Chennai (South)", "lat": 13.0827, "lon": 80.2707},
        {"name": "Pune (West)", "lat": 18.5204, "lon": 73.8567},
        {"name": "Bengaluru (South)", "lat": 12.9716, "lon": 77.5946},
        {"name": "Hyderabad (South)", "lat": 17.3850, "lon": 78.4867},
        {"name": "Jaipur (North-West)", "lat": 26.9124, "lon": 75.7873},
        {"name": "Ahmedabad (West)", "lat": 23.0225, "lon": 72.5714},
        {"name": "Srinagar (North)", "lat": 34.0837, "lon": 74.7973},
        {"name": "Guwahati (North-East)", "lat": 26.1445, "lon": 91.7362},
        {"name": "Kanyakumari (Extreme South)", "lat": 8.0883, "lon": 77.5385}
    ]
    city = np.random.choice(cities)
    offset_lat = np.random.uniform(-0.015, 0.015)
    offset_lon = np.random.uniform(-0.015, 0.015)
    return round(city["lat"] + offset_lat, 6), round(city["lon"] + offset_lon, 6), city["name"]

def get_camera_location(source_key):
    if "video_locations" not in st.session_state:
        st.session_state.video_locations = {}
    
    if source_key not in st.session_state.video_locations:
        lat, lon, city_name = generate_gps()
        digipin_code = digipin_helper.gps_to_digipin(lat, lon)
        st.session_state.video_locations[source_key] = {
            "lat": lat,
            "lon": lon,
            "digipin": digipin_code,
            "city_name": city_name
        }
    return st.session_state.video_locations[source_key]

def randomize_camera_location(source_key):
    if "video_locations" not in st.session_state:
        st.session_state.video_locations = {}
    
    lat, lon, city_name = generate_gps()
    digipin_code = digipin_helper.gps_to_digipin(lat, lon)
    st.session_state.video_locations[source_key] = {
        "lat": lat,
        "lon": lon,
        "digipin": digipin_code,
        "city_name": city_name
    }
