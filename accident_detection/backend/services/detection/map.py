import streamlit as st

# ── Custom Premium Dark Leaflet Cartography HTML ─────────
def get_leaflet_html(lat, lon, digipin, is_incident=False, zoom=14):
    is_dark = (st.session_state.theme == "Cyberpunk Dark")
    
    # Theme configuration
    if is_dark:
        tile_url = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        bg_body = "#06060c"
        bg_popup = "#0b0b14"
        text_color = "#e0e0ea"
        border_color = "#ff3333" if is_incident else "#00ffff"
        color_theme = "#ff3333" if is_incident else "#00ffff"
        popup_title = "🚨 ACTIVE ACCIDENT ANOMALY" if is_incident else "🛰️ BASE STATION MONITOR"
        badge_style = (
            "background: linear-gradient(135deg, #ff8c00, #ff4500); border: 1px solid #ffaa66;"
            if is_incident else
            "background: linear-gradient(135deg, #008080, #004d4d); border: 1px solid #00aaaa;"
        )
        status_msg = (
            '<div class="popup-item" style="color: #ff3333; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">📡 DISPATCHING GPS TRACKER BEACON</div>'
            if is_incident else
            '<div class="popup-item" style="color: #00ffff; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">🟢 PATROL BEACONS SECURE</div>'
        )
    else:
        # Premium Light Mode configuration
        tile_url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
        bg_body = "#f8f9fa"
        bg_popup = "#ffffff"
        text_color = "#0f172a"
        border_color = "#ef4444" if is_incident else "#0d9488"
        color_theme = "#ef4444" if is_incident else "#0d9488"
        popup_title = "🚨 ACTIVE ACCIDENT ANOMALY" if is_incident else "🛰️ BASE STATION MONITOR"
        badge_style = (
            "background: linear-gradient(135deg, #ef4444, #b91c1c); border: 1px solid #fca5a5;"
            if is_incident else
            "background: linear-gradient(135deg, #0d9488, #0f766e); border: 1px solid #99f6e4;"
        )
        status_msg = (
            '<div class="popup-item" style="color: #b91c1c; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">📡 DISPATCHING GPS TRACKER BEACON</div>'
            if is_incident else
            '<div class="popup-item" style="color: #0f766e; font-weight: bold; margin-top: 8px; text-align: center; font-size: 0.8rem; letter-spacing: 0.5px;">🟢 PATROL BEACONS SECURE</div>'
        )
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Aegis Eye Live Geolocation</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body, #map {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                background-color: {bg_body};
            }}
            /* Premium Cyberpunk Leaflet Customizations */
            .leaflet-popup-content-wrapper {{
                background: {bg_popup} !important;
                border: 1px solid {border_color} !important;
                color: {text_color} !important;
                border-radius: 12px !important;
                font-family: monospace !important;
                padding: 8px !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
            }}
            .leaflet-popup-tip {{
                background: {bg_popup} !important;
                border: 1px solid {border_color} !important;
            }}
            .popup-title {{
                color: {color_theme};
                font-weight: 800;
                font-size: 1.05rem;
                letter-spacing: 1px;
                border-bottom: 1px solid rgba(255,255,255,0.08);
                padding-bottom: 6px;
                margin-bottom: 10px;
                text-transform: uppercase;
            }}
            .popup-item {{
                margin: 6px 0;
                font-size: 0.9rem;
                line-height: 1.4;
                color: {text_color} !important;
            }}
            .digipin-badge-popup {{
                display: block;
                {badge_style}
                color: #ffffff !important;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 1rem;
                font-weight: 800;
                text-align: center;
                letter-spacing: 1px;
                margin-top: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            /* Custom Pulsing Target Marker styling */
            .pulsing-marker {{
                position: relative;
            }}
            .pin {{
                width: 12px;
                height: 12px;
                background-color: {color_theme};
                border-radius: 50%;
                border: 2px solid #fff;
                box-shadow: 0 0 10px {color_theme};
                position: absolute;
                top: 6px;
                left: 6px;
            }}
            .pulse {{
                width: 24px;
                height: 24px;
                border: 2px solid {color_theme};
                border-radius: 50%;
                position: absolute;
                top: 0px;
                left: 0px;
                animation: pulsing 1.5s ease-out infinite;
                opacity: 0;
            }}
            @keyframes pulsing {{
                0% {{ transform: scale(0.1); opacity: 0.0; }}
                50% {{ opacity: 0.8; }}
                100% {{ transform: scale(1.3); opacity: 0.0; }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                zoomControl: true,
                attributionControl: false
            }}).setView([{lat}, {lon}], {zoom});

            L.tileLayer('{tile_url}', {{
                maxZoom: 20
            }}).addTo(map);

            var pulsingIcon = L.divIcon({{
                className: 'pulsing-marker',
                html: '<div class="pulse"></div><div class="pin"></div>',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }});

            var marker = L.marker([{lat}, {lon}], {{ icon: pulsingIcon }}).addTo(map);

            var popupContent = `
                <div class="popup-title">{popup_title}</div>
                <div class="popup-item"><b>LATITUDE:</b> {lat:.6f}° N</div>
                <div class="popup-item"><b>LONGITUDE:</b> {lon:.6f}° E</div>
                <div class="popup-item">
                    <b>INDIA POST DIGIPIN:</b>
                    <span class="digipin-badge-popup">🇮🇳 {digipin}</span>
                </div>
                {status_msg}
            `;

            marker.bindPopup(popupContent).openPopup();
        </script>
    </body>
    </html>
    """
    return html
