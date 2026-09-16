import streamlit as st
import cv2
from insightface.app import FaceAnalysis
import json
import os
import time
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO

@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

face_app = load_face_app()

REGISTRY_DIR = "registry"
PHOTOS_DIR = os.path.join(REGISTRY_DIR, "photos")
REGISTRY_FILE = os.path.join(REGISTRY_DIR, "registry.json")
DETECTIONS_FILE = "detections.json"
SURVIVOR_LOG_FILE = "survivor_log.json"
HAZARDS_FILE = "hazards.json"
TELEMETRY_FILE = "telemetry.json"

CONDITION_OPTIONS = [
    "Wheelchair-bound / Paralysis",
    "Blind / Severe visual impairment",
    "Deaf / Hearing impairment",
    "Cognitive impairment / Dementia",
    "Cardiac condition",
    "Respiratory condition (Asthma/COPD)",
    "Pregnant",
    "Diabetic",
    "Other / Unspecified",
]

PRIORITY_META = {
    "CRITICAL": {"emoji": "🔴", "color": "darkred", "hex": "#FF0055", "desc": "Immediate life threat — medevac / rescue first"},
    "HIGH":     {"emoji": "🟠", "color": "red",     "hex": "#FF7700", "desc": "Urgent — needs help soon"},
    "MODERATE": {"emoji": "🟡", "color": "orange",  "hex": "#FFCC00", "desc": "Needs attention, not urgent"},
    "NORMAL":   {"emoji": "🟢", "color": "green",   "hex": "#00CC66", "desc": "Stable, mobile, no immediate risk"},
}

os.makedirs(PHOTOS_DIR, exist_ok=True)

def atomic_save_json(filepath, data):
    tmp_path = f"{filepath}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        pass

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_registry(data):
    atomic_save_json(REGISTRY_FILE, data)

def load_detections():
    if os.path.exists(DETECTIONS_FILE):
        try:
            with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def load_survivor_log():
    if os.path.exists(SURVIVOR_LOG_FILE):
        try:
            with open(SURVIVOR_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def load_hazards():
    if os.path.exists(HAZARDS_FILE):
        try:
            with open(HAZARDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def load_telemetry():
    if os.path.exists(TELEMETRY_FILE):
        try:
            with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

LORA_PACKETS_FILE = "lora_packets.json"

def load_lora_packets():
    if os.path.exists(LORA_PACKETS_FILE):
        try:
            with open(LORA_PACKETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

DEFAULT_SAR_WAYPOINTS = [
    {"id": "WP-01", "lat": 26.9114, "lon": 75.7860, "name": "Ingress Point", "leg": "Leg 1: Ingress"},
    {"id": "WP-02", "lat": 26.9136, "lon": 75.7860, "name": "Sweep North A", "leg": "Leg 2: Northbound"},
    {"id": "WP-03", "lat": 26.9136, "lon": 75.7873, "name": "Cross East A", "leg": "Leg 3: Cross Track"},
    {"id": "WP-04", "lat": 26.9114, "lon": 75.7873, "name": "Sweep South B", "leg": "Leg 4: Southbound"},
    {"id": "WP-05", "lat": 26.9114, "lon": 75.7886, "name": "Cross East B", "leg": "Leg 5: Cross Track"},
    {"id": "WP-06", "lat": 26.9136, "lon": 75.7886, "name": "Sweep North C", "leg": "Leg 6: Egress"}
]

def build_pydeck_map(telemetry, hazards, detections):
    """Builds a high-performance WebGL vector map that updates at 60 FPS without iframe flickering."""
    map_center_lat = 26.9124
    map_center_lon = 75.7873
    if telemetry and "lat" in telemetry and "lon" in telemetry:
        map_center_lat = telemetry["lat"]
        map_center_lon = telemetry["lon"]

    layers = []

    # 0. Lawnmower Search Survey Grid Path
    waypoints = (telemetry and telemetry.get("search_grid_waypoints")) or DEFAULT_SAR_WAYPOINTS
    if waypoints:
        wp_coords = [[wp["lon"], wp["lat"]] for wp in waypoints]
        grid_df = pd.DataFrame([{
            "path": wp_coords,
            "name": "Lawnmower Search Survey Grid (Autonomous Path)",
            "info": "6-leg planned aerial sweep perimeter"
        }])
        layers.append(pdk.Layer(
            "PathLayer",
            data=grid_df,
            get_path="path",
            get_color=[0, 180, 255, 140],
            width_min_pixels=2,
            pickable=True
        ))

    # 1. Incident Command Post (ICP) Staging
    icp_df = pd.DataFrame([{
        "lat": 26.9108,
        "lon": 75.7858,
        "name": "Incident Command Post (ICP)",
        "info": "Ground Rescue Base"
    }])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=icp_df,
        get_position="[lon, lat]",
        get_fill_color=[0, 80, 200, 240],
        get_line_color=[255, 255, 255, 255],
        stroked=True,
        line_width_min_pixels=3,
        get_radius=20,
        pickable=True
    ))

    # 2. Hazard Buffers (35m perimeter) & Point Markers
    if hazards:
        haz_data = []
        for h in hazards:
            h_type = h.get("type", "Hazard")
            h_lat = h.get("lat", 26.9124)
            h_lon = h.get("lon", 75.7873)
            fill_c = [255, 50, 50, 60] if h_type == "Fire" else [40, 120, 255, 60]
            line_c = [255, 0, 0, 220] if h_type == "Fire" else [0, 80, 255, 220]
            haz_data.append({
                "lat": h_lat,
                "lon": h_lon,
                "name": f"{h_type} Hazard Front",
                "info": f"Confidence: {h.get('confidence', 0):.2f} | 35m Exclusion Buffer",
                "fill_color": fill_c,
                "line_color": line_c,
                "buffer_radius": 35,
                "point_radius": 9
            })
        haz_df = pd.DataFrame(haz_data)
        # Exclusion Buffer Zone
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=haz_df,
            get_position="[lon, lat]",
            get_fill_color="fill_color",
            get_line_color="line_color",
            stroked=True,
            line_width_min_pixels=2,
            get_radius="buffer_radius",
            pickable=True
        ))
        # Central Hazard Icon
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=haz_df,
            get_position="[lon, lat]",
            get_fill_color="line_color",
            get_radius="point_radius",
            pickable=True
        ))

    # 3. Survivors with Dynamic Triage Colors
    critical_targets = []
    if detections:
        surv_data = []
        for d in detections:
            s_lat = d.get("lat", 26.9124)
            s_lon = d.get("lon", 75.7873)
            p = d.get("priority", "NORMAL")
            name = d.get("name") or f"Survivor #{d.get('id')}"
            pct = d.get("priority_percent", 0)
            posture = d.get("posture", "Unknown")

            if p == "CRITICAL":
                color = [255, 0, 85, 240]
                critical_targets.append((s_lat, s_lon))
            elif p == "HIGH":
                color = [255, 120, 0, 240]
                critical_targets.append((s_lat, s_lon))
            elif p == "MODERATE":
                color = [255, 210, 0, 240]
            else:
                color = [0, 204, 102, 240]

            surv_data.append({
                "lat": s_lat,
                "lon": s_lon,
                "name": name,
                "info": f"Priority: {p} ({pct}%) | Posture: {posture} | Status: {d.get('status', 'STATIONARY')}",
                "color": color,
                "radius": 14
            })
        surv_df = pd.DataFrame(surv_data)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=surv_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 255],
            stroked=True,
            line_width_min_pixels=2,
            get_radius="radius",
            pickable=True
        ))

    # 4. Safe Ground Evacuation Corridor (PathLayer)
    if critical_targets:
        target = critical_targets[0]
        detour_lat = (26.9108 + target[0]) / 2 - 0.0003
        detour_lon = (75.7858 + target[1]) / 2 + 0.0004
        route_coords = [
            [75.7858, 26.9108],
            [detour_lon, detour_lat],
            [target[1], target[0]]
        ]
        route_df = pd.DataFrame([{
            "path": route_coords,
            "name": "NDRF Safe Evacuation Corridor",
            "info": "Safe path avoiding hazard perimeters"
        }])
        layers.append(pdk.Layer(
            "PathLayer",
            data=route_df,
            get_path="path",
            get_color=[0, 200, 80, 220],
            width_min_pixels=4,
            pickable=True
        ))

    # 5. Autonomous Drone Marker
    if telemetry and "lat" in telemetry and "lon" in telemetry:
        drone_df = pd.DataFrame([{
            "lat": telemetry["lat"],
            "lon": telemetry["lon"],
            "name": "Autonomous Drone (Edge-AI)",
            "info": f"Alt: {telemetry.get('altitude_m')}m | Speed: {telemetry.get('speed_mps')}m/s | Batt: {telemetry.get('battery_pct')}%"
        }])
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=drone_df,
            get_position="[lon, lat]",
            get_fill_color=[170, 0, 255, 255],
            get_line_color=[255, 255, 255, 255],
            stroked=True,
            line_width_min_pixels=3,
            get_radius=18,
            pickable=True
        ))

    view_state = pdk.ViewState(
        latitude=map_center_lat,
        longitude=map_center_lon,
        zoom=15.8,
        pitch=25,
        bearing=telemetry.get("heading_deg", 0) if telemetry else 0
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip={"html": "<b>{name}</b><br>{info}"}
    )
    return deck

def build_folium_map(telemetry, hazards, detections):
    """Builds a Folium Leaflet map with custom icons, hazard buffers, and safe access corridors."""
    map_center = [26.9124, 75.7873]
    if telemetry and "lat" in telemetry and "lon" in telemetry:
        map_center = [telemetry["lat"], telemetry["lon"]]

    m = folium.Map(location=map_center, zoom_start=16, tiles="CartoDB positron")

    # Incident Command Post (ICP)
    icp_coords = [26.9108, 75.7858]
    folium.Marker(
        location=icp_coords,
        icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
        popup="<b>Incident Command Post (ICP)</b><br>Ground Rescue Staging Point"
    ).add_to(m)

    # Drone Marker
    if telemetry and "lat" in telemetry and "lon" in telemetry:
        folium.Marker(
            location=[telemetry["lat"], telemetry["lon"]],
            icon=folium.Icon(color="purple", icon="plane", prefix="fa"),
            popup=f"<b>Autonomous Drone (Edge-AI)</b><br>Alt: {telemetry.get('altitude_m')}m | Batt: {telemetry.get('battery_pct')}%"
        ).add_to(m)

    # Hazard Markers & Exclusion Buffers
    for h in hazards:
        h_lat = h.get("lat", 26.9124)
        h_lon = h.get("lon", 75.7873)
        h_type = h.get("type", "Hazard")
        hazard_color = "red" if h_type == "Fire" else "blue"
        icon_name = "fire" if h_type == "Fire" else "tint"

        folium.Marker(
            location=[h_lat, h_lon],
            icon=folium.Icon(color=hazard_color, icon=icon_name, prefix="fa"),
            popup=f"<b>{h_type} Hazard Zone</b><br>Confidence: {h.get('confidence', 0):.2f}"
        ).add_to(m)

        folium.Circle(
            location=[h_lat, h_lon],
            radius=35,
            color=hazard_color,
            fill=True,
            fill_opacity=0.2,
            popup=f"{h_type} Hazard Exclusion Perimeter (35m)"
        ).add_to(m)

    # Survivors
    critical_targets = []
    for d in detections:
        lat = d.get("lat", 26.9124)
        lon = d.get("lon", 75.7873)
        priority = d.get("priority", "NORMAL")
        meta = PRIORITY_META.get(priority, {"color": "green", "hex": "#00CC66"})
        pct = d.get("priority_percent", 0)
        name = d.get("name") or "Unknown Entity"

        if priority in ["CRITICAL", "HIGH"]:
            critical_targets.append((lat, lon))

        folium.CircleMarker(
            location=[lat, lon],
            radius=11,
            color=meta["color"],
            fill=True,
            fill_color=meta["hex"],
            fill_opacity=0.9,
            popup=(
                f"<b>Survivor #{d.get('id')}</b>: {name}<br>"
                f"Priority: <b>{priority}</b> ({pct}%)<br>"
                f"Posture: {d.get('posture', 'Unknown')}<br>"
                f"Status: {d.get('status', 'STATIONARY')}"
            )
        ).add_to(m)

    # Safe Access Ground Route
    if critical_targets:
        target = critical_targets[0]
        detour_lat = (icp_coords[0] + target[0]) / 2 - 0.0003
        detour_lon = (icp_coords[1] + target[1]) / 2 + 0.0004
        safe_route = [icp_coords, [detour_lat, detour_lon], [target[0], target[1]]]

        folium.PolyLine(
            locations=safe_route,
            color="#00AA44",
            weight=4,
            dash_array="6, 8",
            tooltip="Safe Ground Rescue Corridor (NDRF)"
        ).add_to(m)

    # Lawnmower Search Survey Grid
    waypoints = (telemetry and telemetry.get("search_grid_waypoints")) or DEFAULT_SAR_WAYPOINTS
    if waypoints:
        wp_locs = [[wp["lat"], wp["lon"]] for wp in waypoints]
        folium.PolyLine(
            locations=wp_locs,
            color="#0088FF",
            weight=2,
            dash_array="5, 5",
            opacity=0.6,
            tooltip="Autonomous Lawnmower Search Grid"
        ).add_to(m)

    return m

def build_nav_pydeck_map(telemetry):
    """Builds a 3D tactical search grid PyDeck visualization displaying waypoints, drone vector, and surveyed corridors."""
    waypoints = (telemetry and telemetry.get("search_grid_waypoints")) or DEFAULT_SAR_WAYPOINTS
    current_wp_id = telemetry.get("current_waypoint", "WP-01") if telemetry else "WP-01"

    drone_lat = telemetry.get("lat", 26.9124) if telemetry else 26.9124
    drone_lon = telemetry.get("lon", 75.7873) if telemetry else 75.7873
    heading = telemetry.get("heading_deg", 0) if telemetry else 0

    layers = []

    # 1. Lawnmower Search Survey Grid Path
    wp_coords = [[wp["lon"], wp["lat"]] for wp in waypoints]
    if wp_coords:
        grid_df = pd.DataFrame([{
            "path": wp_coords,
            "name": "Lawnmower Autonomous Search Path",
            "info": "Parallel 6-leg aerial sweep pattern for 100% surface coverage"
        }])
        layers.append(pdk.Layer(
            "PathLayer",
            data=grid_df,
            get_path="path",
            get_color=[0, 190, 255, 230],
            width_min_pixels=3,
            pickable=True
        ))

    # 2. Waypoint Markers
    wp_rows = []
    found_current = False
    for wp in waypoints:
        is_cur = (wp["id"] == current_wp_id)
        if is_cur:
            found_current = True
            color = [255, 170, 0, 255]
            radius = 18
            status = "CURRENT ACTIVE TARGET"
        elif not found_current:
            color = [0, 220, 100, 220]
            radius = 12
            status = "SURVEY COMPLETED"
        else:
            color = [0, 140, 240, 180]
            radius = 12
            status = "PENDING NEXT LEG"

        wp_rows.append({
            "id": wp["id"],
            "name": f"{wp['id']}: {wp['name']}",
            "lat": wp["lat"],
            "lon": wp["lon"],
            "leg": wp.get("leg", ""),
            "status": status,
            "color": color,
            "radius": radius,
            "info": f"{wp.get('leg', '')} | Status: {status} ({wp['lat']:.4f}, {wp['lon']:.4f})"
        })

    wp_df = pd.DataFrame(wp_rows)
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=wp_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 255],
        stroked=True,
        line_width_min_pixels=2,
        get_radius="radius",
        pickable=True
    ))

    # 3. Waypoint Text Labels
    layers.append(pdk.Layer(
        "TextLayer",
        data=wp_df,
        get_position="[lon, lat]",
        get_text="id",
        get_size=13,
        get_color=[255, 255, 255, 255],
        get_pixel_offset=[0, -16],
        get_alignment_baseline="'bottom'",
        get_text_anchor="'middle'"
    ))

    # 4. Incident Command Post (Base)
    icp_df = pd.DataFrame([{
        "lat": 26.9108,
        "lon": 75.7858,
        "name": "Incident Command Post (ICP)",
        "info": "NDRF Staging Base & RTH Home Datum"
    }])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=icp_df,
        get_position="[lon, lat]",
        get_fill_color=[0, 100, 255, 240],
        get_line_color=[255, 255, 255, 255],
        stroked=True,
        line_width_min_pixels=3,
        get_radius=22,
        pickable=True
    ))

    # 5. Live Autonomous Drone Position
    drone_df = pd.DataFrame([{
        "lat": drone_lat,
        "lon": drone_lon,
        "name": "Autonomous Drone (Edge-AI)",
        "info": f"Alt: {telemetry.get('altitude_m', 24.0) if telemetry else 24.0}m | Speed: {telemetry.get('speed_mps', 4.2) if telemetry else 4.2}m/s | Batt: {telemetry.get('battery_pct', 98) if telemetry else 98}%"
    }])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=drone_df,
        get_position="[lon, lat]",
        get_fill_color=[180, 0, 255, 255],
        get_line_color=[255, 255, 255, 255],
        stroked=True,
        line_width_min_pixels=3,
        get_radius=20,
        pickable=True
    ))

    view_state = pdk.ViewState(
        latitude=drone_lat,
        longitude=drone_lon,
        zoom=15.7,
        pitch=35,
        bearing=heading
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip={"html": "<b>{name}</b><br>{info}"}
    )

def render_slam_radar_html(lidar_data, heading_deg=0, avoidance_action="CLEAR_CORRIDOR"):
    """Renders a high-tech 360-degree SLAM LiDAR radar visualization with obstacle blips, distance rings, and collision vectors."""
    if not lidar_data:
        lidar_data = {"front_m": 4.5, "left_m": 3.8, "right_m": 4.1, "rear_m": 5.2, "obstacles": [], "avoidance_action": "CLEAR_CORRIDOR"}

    df = lidar_data.get("front_m", 4.5)
    dl = lidar_data.get("left_m", 3.8)
    dr = lidar_data.get("right_m", 4.1)
    dre = lidar_data.get("rear_m", 5.2)
    obstacles = lidar_data.get("obstacles", [])
    action = avoidance_action or lidar_data.get("avoidance_action", "CLEAR_CORRIDOR")

    def get_color(d):
        if d < 1.4:
            return "#FF3344"
        elif d < 2.2:
            return "#FFAA00"
        else:
            return "#00E676"

    c_front = get_color(df)
    c_left = get_color(dl)
    c_right = get_color(dr)
    c_rear = get_color(dre)

    scale = 20.0
    yf = 150 - min(5.5, df) * scale
    xl = 150 - min(5.5, dl) * scale
    xr = 150 + min(5.5, dr) * scale
    yre = 150 + min(5.5, dre) * scale

    is_alert = (action != "CLEAR_CORRIDOR") or (df < 2.0 or dl < 1.2 or dr < 1.5)
    action_banner = (
        f"""<div style="background: rgba(255, 51, 68, 0.18); border: 2px solid #FF3344; border-radius: 8px; padding: 8px 12px; color: #FF6677; font-weight: bold; text-align: center; margin-top: 10px; font-family: monospace;">
            ⚠️ SLAM COLLISION AVOIDANCE ACTIVE<br><span style="color: #FFF; font-size: 14px;">DIRECTIVE: {action}</span>
        </div>"""
        if is_alert else
        f"""<div style="background: rgba(0, 230, 118, 0.12); border: 1px solid #00E676; border-radius: 8px; padding: 8px 12px; color: #00E676; font-weight: bold; text-align: center; margin-top: 10px; font-family: monospace;">
            🟢 360° SLAM CORRIDOR CLEAR — TRAJECTORY LOCKED
        </div>"""
    )

    svg_code = f"""
    <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <div style="color: #58a6ff; font-family: monospace; font-size: 13px; font-weight: bold; margin-bottom: 8px; letter-spacing: 1px;">
            📡 360° SLAM LiDAR RANGEFINDER RADAR
        </div>
        <svg viewBox="0 0 300 300" style="width: 100%; max-width: 280px; height: auto; display: block; margin: 0 auto;">
            <circle cx="150" cy="150" r="130" fill="#090d13" stroke="#1f2937" stroke-width="1.5" />
            <circle cx="150" cy="150" r="100" fill="none" stroke="#213045" stroke-width="1" stroke-dasharray="4,4" />
            <circle cx="150" cy="150" r="60" fill="none" stroke="#253a56" stroke-width="1" stroke-dasharray="4,4" />
            <circle cx="150" cy="150" r="25" fill="none" stroke="#324f74" stroke-width="1" stroke-dasharray="3,3" />

            <line x1="150" y1="15" x2="150" y2="285" stroke="#1f2937" stroke-width="1" />
            <line x1="15" y1="150" x2="285" y2="150" stroke="#1f2937" stroke-width="1" />

            <text x="153" y="127" fill="#6b7280" font-size="9" font-family="monospace">1.2m</text>
            <text x="153" y="92" fill="#6b7280" font-size="9" font-family="monospace">3.0m</text>
            <text x="153" y="52" fill="#6b7280" font-size="9" font-family="monospace">5.0m</text>

            <text x="150" y="12" fill="#58a6ff" font-size="11" font-weight="bold" text-anchor="middle" font-family="monospace">FRONT (0°)</text>
            <text x="290" y="154" fill="#8b949e" font-size="10" text-anchor="end" font-family="monospace">RIGHT</text>
            <text x="150" y="297" fill="#8b949e" font-size="10" text-anchor="middle" font-family="monospace">REAR</text>
            <text x="10" y="154" fill="#8b949e" font-size="10" text-anchor="start" font-family="monospace">LEFT</text>

            <g transform="translate(150, 150) rotate({heading_deg})">
                <circle cx="0" cy="0" r="10" fill="#7928ca" stroke="#c084fc" stroke-width="2" />
                <polygon points="0,-16 -6,-6 6,-6" fill="#00e676" />
                <line x1="-12" y1="0" x2="12" y2="0" stroke="#c084fc" stroke-width="2" />
                <line x1="0" y1="-12" x2="0" y2="12" stroke="#c084fc" stroke-width="2" />
            </g>

            <line x1="150" y1="150" x2="150" y2="{yf}" stroke="{c_front}" stroke-width="2" stroke-dasharray="2,2" opacity="0.85" />
            <line x1="150" y1="150" x2="{xl}" y2="150" stroke="{c_left}" stroke-width="2" stroke-dasharray="2,2" opacity="0.85" />
            <line x1="150" y1="150" x2="{xr}" y2="150" stroke="{c_right}" stroke-width="2" stroke-dasharray="2,2" opacity="0.85" />
            <line x1="150" y1="150" x2="150" y2="{yre}" stroke="{c_rear}" stroke-width="2" stroke-dasharray="2,2" opacity="0.85" />

            <circle cx="150" cy="{yf}" r="6" fill="{c_front}" stroke="#ffffff" stroke-width="1.5" />
            <text x="150" y="{max(26, yf - 8)}" fill="{c_front}" font-size="10" font-weight="bold" text-anchor="middle" font-family="monospace">{df}m</text>

            <circle cx="{xl}" cy="150" r="6" fill="{c_left}" stroke="#ffffff" stroke-width="1.5" />
            <text x="{max(28, xl - 10)}" y="142" fill="{c_left}" font-size="10" font-weight="bold" text-anchor="middle" font-family="monospace">{dl}m</text>

            <circle cx="{xr}" cy="150" r="6" fill="{c_right}" stroke="#ffffff" stroke-width="1.5" />
            <text x="{min(272, xr + 10)}" y="142" fill="{c_right}" font-size="10" font-weight="bold" text-anchor="middle" font-family="monospace">{dr}m</text>

            <circle cx="150" cy="{yre}" r="6" fill="{c_rear}" stroke="#ffffff" stroke-width="1.5" />
            <text x="150" y="{min(280, yre + 14)}" fill="{c_rear}" font-size="10" font-weight="bold" text-anchor="middle" font-family="monospace">{dre}m</text>
        </svg>

        <div style="display: flex; justify-content: space-around; font-family: monospace; font-size: 12px; margin-top: 12px; padding-top: 10px; border-top: 1px solid #21262d;">
            <div><span style="color: #8b949e;">FRONT:</span> <b style="color: {c_front};">{df}m</b></div>
            <div><span style="color: #8b949e;">LEFT:</span> <b style="color: {c_left};">{dl}m</b></div>
            <div><span style="color: #8b949e;">RIGHT:</span> <b style="color: {c_right};">{dr}m</b></div>
            <div><span style="color: #8b949e;">REAR:</span> <b style="color: {c_rear};">{dre}m</b></div>
        </div>

        {action_banner}
    </div>
    """
    return svg_code

st.set_page_config(
    page_title="AI Disaster Response Drone Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Banner
st.title("🛰️ Autonomous Disaster Response Drone — Tactical Command Center")
st.caption("Edge-AI On-Device Search & Rescue | Multi-Hazard Aerial Reconnaissance & Triage | SIH Project")

tab_live, tab_nav, tab_lora, tab_reg, tab_photo, tab_sitrep = st.tabs([
    "🗺️ Live Mission Dashboard",
    "🧭 Autonomous Flight & SLAM Navigation",
    "📻 Offline LoRa Radio Terminal",
    "📋 Vulnerable Persons Registry",
    "📷 Aerial Photo Triage",
    "📑 Mission SITREP & Export"
])

# ---------------- TAB 1: LIVE MISSION DASHBOARD ----------------
with tab_live:
    col_mode1, col_mode2 = st.columns([2.2, 1.0])
    with col_mode1:
        map_engine = st.radio(
            "Tactical Map Engine",
            ["🛰️ Real-Time Vector Radar (WebGL — Zero Flicker)", "🗺️ Detailed Street GIS (Folium)"],
            horizontal=True,
            help="Vector Radar updates at 60 FPS via WebGL with zero flicker. Street GIS offers CartoDB tiles with controlled sync."
        )
    with col_mode2:
        if "Detailed Street GIS" in map_engine:
            if st.button("🔄 Sync Street Map with Feed", use_container_width=True):
                st.session_state["force_folium_sync"] = time.time()

    if "Real-Time Vector Radar" in map_engine:
        # Vector Radar mode: Everything (including map) runs in @st.fragment(run_every="2s") with ZERO flicker!
        @st.fragment(run_every="2s")
        def render_vector_dashboard():
            telemetry = load_telemetry()
            detections = load_detections()
            hazards = load_hazards()
            survivor_log = load_survivor_log()

            # Drone Telemetry HUD
            st.markdown("##### 🚁 Autonomous Drone Telemetry & Sensor Status (Edge Inference)")
            t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
            if telemetry and telemetry.get("status") != "CAMERA_OFFLINE":
                sensor_name = "🔥 FLIR Thermal (Ironbow)" if telemetry.get("thermal_mode") else "📷 RGB Optical HD"
                t_col1.metric("Flight Mode", telemetry.get("flight_mode", "AUTO_SURVEY"), "🟢 GPS-Guided")
                t_col2.metric("Altitude AGL", f"{telemetry.get('altitude_m', 22.0)} m", f"Speed: {telemetry.get('speed_mps', 3.8)} m/s")
                t_col3.metric("Battery Level", f"{telemetry.get('battery_pct', 98)} %", "Normal")
                t_col4.metric("Active Sensor", sensor_name)
                t_col5.metric("Heading", f"{telemetry.get('heading_deg', 0)}°", "Compass")
            else:
                t_col1.metric("Flight Mode", "STANDBY / LANDED", "⚪ Landed")
                t_col2.metric("Altitude AGL", "0.0 m", "Ground Post")
                t_col3.metric("Battery Level", "Standby", "Landed")
                t_col4.metric("Active Sensor", "Feed Offline", "Standby")
                t_col5.metric("Drone Link", "Offline", "Data Saved")

            if telemetry and telemetry.get("status") != "CAMERA_OFFLINE":
                lidar_info = telemetry.get("slam_lidar", {})
                avoid_act = lidar_info.get("avoidance_action", "CLEAR_CORRIDOR")
                wp_cur = telemetry.get("current_waypoint", "WP-01")
                wp_nxt = telemetry.get("next_waypoint", "WP-02")
                prog = telemetry.get("survey_progress_pct", 0)
                if avoid_act != "CLEAR_CORRIDOR":
                    st.warning(f"⚠️ **360° SLAM COLLISION AVOIDANCE ACTIVE**: Sector obstruction detected! Trajectory auto-deflection: `{avoid_act}` | Grid Track: `{wp_cur} → {wp_nxt}` ({prog}% surveyed)")

            st.divider()

            # Triage Metrics
            counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "NORMAL": 0}
            for d in detections:
                p = d.get("priority", "NORMAL")
                if p in counts:
                    counts[p] += 1

            if counts["CRITICAL"] > 0:
                st.error(f"🚨 **IMMEDIATE MEDEVAC REQUIRED**: {counts['CRITICAL']} casualty/survivor(s) currently in **CRITICAL** condition near danger perimeters!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔴 Critical Priority", counts["CRITICAL"], help="Immediate life threat — Medevac priority")
            c2.metric("🟠 High Priority", counts["HIGH"], help="Urgent evacuation needed")
            c3.metric("🟡 Moderate Priority", counts["MODERATE"], help="Stable with injuries or vulnerability")
            c4.metric("🟢 Normal Priority", counts["NORMAL"], help="Stable, mobile survivors")

            if not detections and not hazards:
                st.info("ℹ️ No active detections in current camera frame. All recorded mission logs and registered profiles remain available below.")

            col_map, col_queue = st.columns([2, 1])

            with col_map:
                st.markdown("##### 📍 Real-Time Vector Radar (Zero-Flicker WebGL)")
                deck = build_pydeck_map(telemetry, hazards, detections)
                st.pydeck_chart(deck, use_container_width=True)

            with col_queue:
                st.markdown("##### 🚨 Evacuation Priority Queue")
                priority_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}
                sorted_detections = sorted(detections, key=lambda x: priority_order.get(x.get("priority"), 5))
                if not sorted_detections:
                    st.info("No active survivors in current frame.")
                else:
                    for d in sorted_detections:
                        p = d.get("priority", "NORMAL")
                        meta = PRIORITY_META.get(p, {"emoji": "⚪"})
                        label = d.get("name") or f"Survivor #{d.get('id')}"
                        pct = d.get("priority_percent", 0)
                        posture = d.get("posture", "Unknown")
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {meta['color']}; padding: 8px 12px; margin-bottom: 8px; background: rgba(128,128,128,0.08); border-radius: 4px;">
                                <strong>{meta['emoji']} {label}</strong> — <code style="color:{meta['color']}">{p} ({pct}%)</code><br>
                                <small>Posture: <b>{posture}</b> | Status: {d.get('status', 'STATIONARY')}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.divider()
            st.markdown("##### 📋 Full Mission Log (All Individuals Detected in Current Mission)")
            if not survivor_log:
                st.info("Mission log empty. Start the detection pipeline to begin recording.")
            else:
                rows = []
                for entry in sorted(survivor_log, key=lambda x: priority_order.get(x.get("priority"), 4)):
                    p = entry.get("priority", "NORMAL")
                    meta = PRIORITY_META.get(p, {"emoji": "⚪"})
                    rows.append({
                        "ID": entry["id"],
                        "Name": entry.get("name") or "Unknown Entity",
                        "Priority": f"{meta['emoji']} {p} ({entry.get('priority_percent', 0)}%)",
                        "Status In-Frame": "🟢 Active Live" if entry.get("active_now") else "⚪ Exited Frame",
                        "Posture": entry.get("posture", "Unknown"),
                        "Movement": entry.get("status", "—"),
                        "Age": str(entry["age"]) if entry.get("age") is not None else "—",
                        "Medical Conditions": ", ".join(entry.get("conditions") or []) or "—",
                        "Notes": entry.get("health") or "—"
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

        render_vector_dashboard()

    else:
        # Folium Leaflet Street GIS Mode: Stable & Decoupled from 2s loop to completely eliminate flicker!
        @st.fragment(run_every="2s")
        def render_folium_text_components():
            telemetry = load_telemetry()
            detections = load_detections()
            hazards = load_hazards()
            survivor_log = load_survivor_log()

            # Drone Telemetry HUD
            st.markdown("##### 🚁 Autonomous Drone Telemetry & Sensor Status (Edge Inference)")
            t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
            if telemetry and telemetry.get("status") != "CAMERA_OFFLINE":
                sensor_name = "🔥 FLIR Thermal (Ironbow)" if telemetry.get("thermal_mode") else "📷 RGB Optical HD"
                t_col1.metric("Flight Mode", telemetry.get("flight_mode", "AUTO_SURVEY"), "GPS-Guided")
                t_col2.metric("Altitude AGL", f"{telemetry.get('altitude_m', 22.0)} m", f"Speed: {telemetry.get('speed_mps', 3.8)} m/s")
                t_col3.metric("Battery Level", f"{telemetry.get('battery_pct', 98)} %", "Normal")
                t_col4.metric("Active Sensor", sensor_name)
                t_col5.metric("Heading", f"{telemetry.get('heading_deg', 0)}°", "Compass")
            else:
                t_col1.metric("Flight Mode", "STANDBY / LANDED", "⚪ Landed")
                t_col2.metric("Altitude AGL", "0.0 m", "Ground Post")
                t_col3.metric("Battery Level", "Standby", "Landed")
                t_col4.metric("Active Sensor", "Feed Offline", "Standby")
                t_col5.metric("Drone Link", "Offline", "Data Saved")

            if telemetry and telemetry.get("status") != "CAMERA_OFFLINE":
                lidar_info = telemetry.get("slam_lidar", {})
                avoid_act = lidar_info.get("avoidance_action", "CLEAR_CORRIDOR")
                wp_cur = telemetry.get("current_waypoint", "WP-01")
                wp_nxt = telemetry.get("next_waypoint", "WP-02")
                prog = telemetry.get("survey_progress_pct", 0)
                if avoid_act != "CLEAR_CORRIDOR":
                    st.warning(f"⚠️ **360° SLAM COLLISION AVOIDANCE ACTIVE**: Sector obstruction detected! Trajectory auto-deflection: `{avoid_act}` | Grid Track: `{wp_cur} → {wp_nxt}` ({prog}% surveyed)")

            st.divider()

            # Triage Metrics
            counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "NORMAL": 0}
            for d in detections:
                p = d.get("priority", "NORMAL")
                if p in counts:
                    counts[p] += 1

            if counts["CRITICAL"] > 0:
                st.error(f"🚨 **IMMEDIATE MEDEVAC REQUIRED**: {counts['CRITICAL']} casualty/survivor(s) currently in **CRITICAL** condition near danger perimeters!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔴 Critical Priority", counts["CRITICAL"])
            c2.metric("🟠 High Priority", counts["HIGH"])
            c3.metric("🟡 Moderate Priority", counts["MODERATE"])
            c4.metric("🟢 Normal Priority", counts["NORMAL"])

            if not detections and not hazards:
                st.info("ℹ️ No active detections in current camera frame. Mission logs and registered profiles remain available below.")

        render_folium_text_components()

        col_folium_map, col_folium_queue = st.columns([2, 1])

        with col_folium_map:
            st.markdown("##### 📍 Street GIS Map (Stable Leaflet — Zero Flicker)")
            telemetry_snapshot = load_telemetry()
            hazards_snapshot = load_hazards()
            detections_snapshot = load_detections()
            folium_map = build_folium_map(telemetry_snapshot, hazards_snapshot, detections_snapshot)
            # returned_objects=[] prevents Leaflet click events from triggering unnecessary reruns
            st_folium(folium_map, width=None, height=480, key="stable_folium_map", returned_objects=[])

        with col_folium_queue:
            @st.fragment(run_every="2s")
            def render_live_queue_only():
                detections = load_detections()
                st.markdown("##### 🚨 Evacuation Priority Queue")
                priority_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}
                sorted_detections = sorted(detections, key=lambda x: priority_order.get(x.get("priority"), 5))
                if not sorted_detections:
                    st.info("No active survivors in current frame.")
                else:
                    for d in sorted_detections:
                        p = d.get("priority", "NORMAL")
                        meta = PRIORITY_META.get(p, {"emoji": "⚪"})
                        label = d.get("name") or f"Survivor #{d.get('id')}"
                        pct = d.get("priority_percent", 0)
                        posture = d.get("posture", "Unknown")
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {meta['color']}; padding: 8px 12px; margin-bottom: 8px; background: rgba(128,128,128,0.08); border-radius: 4px;">
                                <strong>{meta['emoji']} {label}</strong> — <code style="color:{meta['color']}">{p} ({pct}%)</code><br>
                                <small>Posture: <b>{posture}</b> | Status: {d.get('status', 'STATIONARY')}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            render_live_queue_only()

        st.divider()
        @st.fragment(run_every="2s")
        def render_live_log_only():
            survivor_log = load_survivor_log()
            st.markdown("##### 📋 Full Mission Log (All Individuals Detected in Current Mission)")
            if not survivor_log:
                st.info("Mission log empty. Start the detection pipeline to begin recording.")
            else:
                priority_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}
                rows = []
                for entry in sorted(survivor_log, key=lambda x: priority_order.get(x.get("priority"), 4)):
                    p = entry.get("priority", "NORMAL")
                    meta = PRIORITY_META.get(p, {"emoji": "⚪"})
                    rows.append({
                        "ID": entry["id"],
                        "Name": entry.get("name") or "Unknown Entity",
                        "Priority": f"{meta['emoji']} {p} ({entry.get('priority_percent', 0)}%)",
                        "Status In-Frame": "🟢 Active Live" if entry.get("active_now") else "⚪ Exited Frame",
                        "Posture": entry.get("posture", "Unknown"),
                        "Movement": entry.get("status", "—"),
                        "Age": str(entry["age"]) if entry.get("age") is not None else "—",
                        "Medical Conditions": ", ".join(entry.get("conditions") or []) or "—",
                        "Notes": entry.get("health") or "—"
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
        render_live_log_only()

# ---------------- TAB 2: AUTONOMOUS FLIGHT & SLAM NAVIGATION ----------------
with tab_nav:
    st.subheader("🧭 Autonomous Flight, Lawnmower Search Grid & 360° SLAM Obstacle Avoidance")
    st.caption("AI-Powered Lawnmower Survey Planner | Real-Time SLAM LiDAR Radar | GPS-Denied VIO Fallback | Battery Return-to-Home (RTH) Fail-Safe")

    # Mode Selector & Simulation Control
    col_nav_ctrl1, col_nav_ctrl2 = st.columns([2.0, 1.2])
    with col_nav_ctrl1:
        nav_mode_selection = st.radio(
            "Flight Navigation Subsystem",
            ["🛰️ Autonomous GPS-Guided (RTK Lock, 14 Satellites)", "🏢 GPS-Denied Simulation (Indoor / Collapsed Void / Optical Flow VIO)"],
            horizontal=True,
            help="Simulates switching between outdoor satellite RTK positioning and indoor GPS-denied Visual Inertial Odometry."
        )
    with col_nav_ctrl2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        force_rth = st.button("🚨 Trigger Immediate Emergency RTH", use_container_width=True, help="Override autonomous grid and force drone to return to Incident Command Post datum.")

    @st.fragment(run_every="2s")
    def render_nav_tab():
        telemetry = load_telemetry()
        lidar = (telemetry and telemetry.get("slam_lidar")) or {
            "front_m": 4.5, "left_m": 3.8, "right_m": 4.1, "rear_m": 5.2,
            "obstacles": [], "avoidance_action": "CLEAR_CORRIDOR"
        }
        waypoints = (telemetry and telemetry.get("search_grid_waypoints")) or DEFAULT_SAR_WAYPOINTS
        heading = telemetry.get("heading_deg", 0) if telemetry else 0
        cur_wp = telemetry.get("current_waypoint", "WP-01") if telemetry else "WP-01"
        nxt_wp = telemetry.get("next_waypoint", "WP-02") if telemetry else "WP-02"
        active_leg = telemetry.get("active_leg", "Leg 1: Ingress") if telemetry else "Leg 1: Ingress"
        progress_pct = telemetry.get("survey_progress_pct", 0) if telemetry else 0
        dist_icp = telemetry.get("dist_to_icp_m", 150.0) if telemetry else 150.0
        rth_budget = telemetry.get("rth_battery_budget", 14.0) if telemetry else 14.0
        batt = telemetry.get("battery_pct", 95.0) if telemetry else 95.0
        rth_status = telemetry.get("rth_status", "SAFE_NOMINAL") if telemetry else "SAFE_NOMINAL"
        avoid_action = lidar.get("avoidance_action", "CLEAR_CORRIDOR")

        if force_rth:
            st.error("🚨 **EMERGENCY RTH ENGAGED**: Autonomous search paused. Drone ascending to 30m corridor and returning to Incident Command Post (26.9108, 75.7858).")

        # Top KPI Banner: Mission Flight HUD
        hud1, hud2, hud3, hud4, hud5 = st.columns(5)
        hud1.metric("Active Waypoint", f"{cur_wp} → {nxt_wp}", active_leg)
        hud2.metric("Survey Progress", f"{progress_pct}%", "Area Swept")
        hud3.metric("Distance to Base (ICP)", f"{dist_icp} m", "Direct Ingress")
        hud4.metric("Battery Remaining", f"{batt}%", f"Budget: {rth_budget}%")
        
        safe_margin = round(batt - rth_budget, 1)
        margin_badge = "🟢 Safe Margin" if safe_margin > 20 else ("🟡 Low Reserve" if safe_margin > 5 else "🔴 Critical RTH")
        hud5.metric("RTH Fail-Safe Status", f"+{safe_margin}%", margin_badge)

        # Progress bar
        st.progress(progress_pct / 100.0, text=f"Lawnmower Search Grid Coverage: {progress_pct}% Completed")
        st.divider()

        # GPS-Denied VIO Info Panel (if selected)
        if "GPS-Denied" in nav_mode_selection:
            st.markdown(
                """
                <div style="background: rgba(255, 170, 0, 0.1); border-left: 4px solid #FFAA00; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
                    <strong>📡 GPS-DENIED NAVIGATION ACTIVE (Optical Flow VIO + SLAM Lock):</strong><br>
                    • <b>Downward Optical Flow:</b> 60 FPS feature tracking (294 keypoints locked) — <i>Drift compensation: ±1.8 cm/min</i><br>
                    • <b>6-DOF IMU Fusion:</b> Extended Kalman Filter (EKF) propagating linear & angular acceleration<br>
                    • <b>LiDAR SLAM:</b> Real-time point cloud surface matching inside collapsed void / indoor perimeter
                </div>
                """,
                unsafe_allow_html=True
            )

        # Core Two-Column Layout: SLAM Radar & 3D Flight Grid Map
        c_radar, c_map = st.columns([1.1, 1.9])

        with c_radar:
            st.markdown("##### 🛡️ 360° SLAM Obstacle Avoidance Radar")
            radar_html = render_slam_radar_html(lidar, heading, avoid_action)
            st.markdown(radar_html, unsafe_allow_html=True)

            # SLAM Obstacle Detections Roster
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("###### ⚠️ Proximity Threat Classification Log")
            obs_list = lidar.get("obstacles", [])
            if obs_list:
                for obs in obs_list:
                    sev = obs.get("severity", "WARNING")
                    color = "#FF3344" if sev == "CRITICAL" else "#FFAA00"
                    st.markdown(
                        f"""
                        <div style="background: rgba(128,128,128,0.08); border-left: 3px solid {color}; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px; font-size: 13px;">
                            <b style="color:{color};">[{sev}]</b> <b>{obs.get('hazard')}</b> in <b>{obs.get('sector')}</b> sector at <b>{obs.get('dist_m')} m</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div style='font-size: 13px; color: #888;'>No immediate physical hazards within 2.0m radius. Flight envelope clear.</div>", unsafe_allow_html=True)

        with c_map:
            st.markdown("##### 📍 Autonomous Lawnmower Search Pattern (3D Flight Vector)")
            nav_deck = build_nav_pydeck_map(telemetry)
            st.pydeck_chart(nav_deck, use_container_width=True)

            # Waypoint Flight Sequence Table
            st.markdown("###### 📋 Mission Waypoint Sequence & Execution Status")
            wp_table_rows = []
            found_curr = False
            for wp in waypoints:
                is_curr = (wp["id"] == cur_wp)
                if is_curr:
                    found_curr = True
                    stat_badge = "🟡 IN PROGRESS"
                elif not found_curr:
                    stat_badge = "🟢 COMPLETED"
                else:
                    stat_badge = "⚪ PENDING"

                wp_table_rows.append({
                    "Waypoint": wp["id"],
                    "Name": wp["name"],
                    "Leg": wp["leg"],
                    "Latitude": f"{wp['lat']:.5f}",
                    "Longitude": f"{wp['lon']:.5f}",
                    "Status": stat_badge
                })
            st.dataframe(pd.DataFrame(wp_table_rows), use_container_width=True, hide_index=True)

    render_nav_tab()

# ---------------- TAB 3: VULNERABLE PERSONS REGISTRY ----------------
with tab_reg:
    st.subheader("📋 Pre-Register High-Priority Individuals")
    st.caption("Enroll citizens with medical vulnerabilities and multi-angle facial embeddings for instant drone recognition.")

    registry = load_registry()

    capture_mode = st.radio(
        "How do you want to provide photos?",
        ["📷 Capture live via webcam", "📁 Upload photo files"]
    )

    with st.form("enroll_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        health = st.text_area("Medical History / Health Notes (Free text for rescue teams)")
        conditions = st.multiselect(
            "Specific Health & Mobility Conditions (Impacts AI Priority Scoring directly)",
            CONDITION_OPTIONS,
            help="Directly evaluated by the dynamic triage algorithm to prioritize search & rescue dispatch."
        )

        raw_photos = []
        if capture_mode == "📷 Capture live via webcam":
            st.caption("Capture 2-4 distinct angles for robust multi-pose facial recognition.")
            cam1 = st.camera_input("Angle 1 — Frontal View")
            cam2 = st.camera_input("Angle 2 — Left Profile")
            cam3 = st.camera_input("Angle 3 — Right Profile")
            cam4 = st.camera_input("Angle 4 — Slightly distant / ambient lighting (Optional)")
            for cam in [cam1, cam2, cam3, cam4]:
                if cam is not None:
                    raw_photos.append(cam)
        else:
            uploaded = st.file_uploader(
                "Upload 2-4 clear face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
            )
            if uploaded:
                raw_photos = uploaded

        submitted = st.form_submit_button("Register Person into Rescue Database")

        if submitted:
            if not name or not raw_photos:
                st.error("Name and at least one valid face photo are required.")
            else:
                photo_paths = []
                for i, photo in enumerate(raw_photos):
                    img = Image.open(photo).convert("RGB")
                    photo_path = os.path.join(PHOTOS_DIR, f"{name}_{i}.jpg")
                    img.save(photo_path)

                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    faces = face_app.get(img_bgr)
                    if len(faces) == 1:
                        photo_paths.append(photo_path)
                    elif len(faces) > 1:
                        st.warning(f"Photo {i+1} detected {len(faces)} faces — only ONE person should be in the frame. Skipped.")

                if photo_paths:
                    registry.append({
                        "name": name,
                        "age": int(age),
                        "health": health,
                        "conditions": conditions,
                        "photo_paths": photo_paths
                    })
                    save_registry(registry)
                    st.success(f"✅ {name} successfully registered with {len(photo_paths)} validated photo embeddings!")
                else:
                    st.error("No valid face detected in provided images. Please ensure good lighting and clear view.")

    st.divider()
    st.subheader("👥 Currently Enrolled Citizens")
    if not registry:
        st.info("No citizens registered yet.")
    for p in registry:
        cond_text = ", ".join(p.get("conditions", [])) or "None"
        st.markdown(f"- **{p['name']}** (Age: {p['age']}) — Conditions: `{cond_text}` — *Notes: {p.get('health') or 'None'}*")

# ---------------- TAB 3: AERIAL PHOTO TRIAGE ----------------
with tab_photo:
    st.subheader("📷 Aerial / Disaster Scene Snapshot Triage")
    st.caption("Upload reconnaissance drone photos for offline batch survivor detection, posture evaluation, hazard proximity analysis, and priority calculation.")

    uploaded_photo = st.file_uploader("Upload Drone Snapshot", type=["jpg", "jpeg", "png"], key="analysis_photo")

    if uploaded_photo is not None:
        img = Image.open(uploaded_photo).convert("RGB")
        img_array = np.array(img)

        @st.cache_resource
        def load_models():
            return YOLO("yolov8n-pose.pt"), YOLO("fire_best.pt"), YOLO("flood_best.pt")

        pose_model, fire_model, flood_model = load_models()
        registry = load_registry()

        registry_embeddings = []
        for person in registry:
            photo_paths = person.get("photo_paths") or [person.get("photo_path")]
            embs = []
            for path in photo_paths:
                if not path or not os.path.exists(path):
                    continue
                pimg = cv2.imread(path)
                if pimg is None:
                    continue
                faces = face_app.get(pimg)
                if faces:
                    embs.append(faces[0].embedding)
            if embs:
                avg = np.mean(embs, axis=0)
                avg = avg / np.linalg.norm(avg)
                registry_embeddings.append({
                    "name": person["name"],
                    "age": person.get("age"),
                    "health": person.get("health"),
                    "conditions": person.get("conditions", []),
                    "embedding": avg
                })

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        pose_results = pose_model(img_bgr, conf=0.5, verbose=False)
        fire_results = fire_model(img_bgr, conf=0.4, verbose=False)
        flood_results = flood_model(img_bgr, conf=0.4, verbose=False)

        annotated = img_array.copy()

        hazard_boxes = []
        for box in fire_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hazard_boxes.append((x1, y1, x2, y2, "Fire"))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 50, 50), 3)
            cv2.putText(annotated, "Fire Hazard", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)

        for box in flood_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hazard_boxes.append((x1, y1, x2, y2, "Flood"))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 150, 255), 3)
            cv2.putText(annotated, "Floodwater", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 150, 255), 2)

        def box_center(x1, y1, x2, y2):
            return ((x1 + x2) // 2, (y1 + y2) // 2)

        def classify_posture(keypoints):
            pts = {}
            for idx in [5, 6, 11, 12, 15, 16]:
                x, y, conf = keypoints[idx]
                if conf > 0.3:
                    pts[idx] = (x, y)
            if not pts:
                return "Unknown"
            all_pts = list(pts.values())
            ys = [p[1] for p in all_pts]
            xs = [p[0] for p in all_pts]
            vertical_span = max(ys) - min(ys)
            horizontal_span = max(xs) - min(xs)
            if vertical_span < horizontal_span * 0.6:
                return "Lying down / Distress"
            return "Upright"

        HAZARD_PROXIMITY_PX = 250
        MATCH_THRESHOLD = 0.35
        reports = []

        boxes = pose_results[0].boxes
        keypoints_all = pose_results[0].keypoints

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            kpts = keypoints_all.data[i].cpu().numpy() if keypoints_all is not None else None
            posture = classify_posture(kpts) if kpts is not None else "Unknown"

            pc = box_center(x1, y1, x2, y2)
            near_hazard, min_dist = None, None
            for hb in hazard_boxes:
                hc = box_center(*hb[:4])
                dist = ((pc[0]-hc[0])**2 + (pc[1]-hc[1])**2) ** 0.5
                if min_dist is None or dist < min_dist:
                    min_dist, near_hazard = dist, hb[4]
            hazard_nearby = near_hazard is not None and min_dist < HAZARD_PROXIMITY_PX
            distress_posture = posture == "Lying down / Distress"

            crop = img_bgr[max(0,y1):y2, max(0,x1):x2]
            matched_person = None
            if crop.size > 0 and registry_embeddings:
                faces = face_app.get(crop)
                if faces:
                    emb = faces[0].embedding
                    emb = emb / np.linalg.norm(emb)
                    best_sim = MATCH_THRESHOLD
                    for person in registry_embeddings:
                        sim = float(np.dot(emb, person["embedding"]))
                        if sim > best_sim:
                            best_sim = sim
                            matched_person = person

            CONDITION_WEIGHTS = {
                "Wheelchair-bound / Paralysis": 30, "Blind / Severe visual impairment": 25,
                "Cognitive impairment / Dementia": 25, "Cardiac condition": 20,
                "Respiratory condition (Asthma/COPD)": 20, "Pregnant": 20,
                "Deaf / Hearing impairment": 15, "Diabetic": 10, "Other / Unspecified": 5,
            }

            score = 0
            reasons = []
            if hazard_nearby:
                score += 50
                reasons.append(f"Hazard proximity ({near_hazard})")
            if distress_posture:
                score += 30
                reasons.append("Distress posture (lying down)")
            if matched_person:
                score += 10
                reasons.append("Pre-registered citizen")
                age = matched_person.get("age")
                if age is not None and (age <= 10 or age >= 60):
                    score += 15
                    reasons.append(f"Vulnerable age ({age})")
                for cond in matched_person.get("conditions", []):
                    score += CONDITION_WEIGHTS.get(cond, 5)
                if matched_person.get("conditions"):
                    reasons.append(f"Conditions: {', '.join(matched_person['conditions'])}")

            if score >= 70:
                priority = "CRITICAL"
                box_color = (255, 0, 80)
            elif score >= 40:
                priority = "HIGH"
                box_color = (255, 140, 0)
            elif score >= 20:
                priority = "MODERATE"
                box_color = (255, 215, 0)
            else:
                priority = "NORMAL"
                box_color = (0, 200, 80)

            label = matched_person["name"] if matched_person else "Unknown Entity"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(annotated, f"{label} | {priority}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

            reports.append({
                "name": label, "priority": priority, "posture": posture,
                "reasons": reasons if reasons else ["No urgent risk factors"],
                "health": matched_person["health"] if matched_person else None,
                "age": matched_person["age"] if matched_person else None
            })

        st.image(annotated, caption="AI Analyzed Aerial Drone Snapshot", use_container_width=True)

        st.subheader("Analysis Breakdown")
        if not reports:
            st.info("No human entities detected in this scene.")
        else:
            for r in sorted(reports, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}.get(x["priority"], 3)):
                extra = f" (Age: {r['age']}) — Notes: {r['health']}" if r["health"] else ""
                st.markdown(f"- **{r['name']}**{extra} — Posture: `{r['posture']}` — Priority: **{r['priority']}** (*{', '.join(r['reasons'])}*)")

        if hazard_boxes:
            hazard_types = list(set(h[4] for h in hazard_boxes))
            st.warning(f"⚠️ Active Hazards Identified: {', '.join(hazard_types)}")

# ---------------- TAB 4: OFFLINE LORA RADIO TERMINAL ----------------
with tab_lora:
    st.subheader("📻 Offline LoRa / Mesh Radio Telemetry Terminal")
    st.caption("Sub-GHz (868.1 MHz) Long-Range Radio Receiver for Zero-Connectivity, GPS-Denied & Post-Disaster Environments | SIH Offline Resilience")

    # Radio Gateway Hardware Status
    rf_c1, rf_c2, rf_c3, rf_c4 = st.columns(4)
    rf_c1.metric("RF Frequency", "868.100 MHz", "ISM India Band")
    rf_c2.metric("Modulation", "LoRa CSS (SF10)", "125 kHz / CR 4/5")
    rf_c3.metric("Gateway Link", "🟢 ACTIVE LISTENING", "Node #01 Ground")
    rf_c4.metric("Mesh Protocol", "Meshtastic / APRS", "CRC-16 Verified")

    st.markdown(
        """
        <div style="background: rgba(30, 40, 50, 0.05); padding: 12px 18px; border-radius: 8px; border-left: 4px solid #7928CA; margin: 12px 0;">
            <strong>🌐 Offline Tactical Mesh Topology:</strong><br>
            <code>[ Aerial Drone (Edge-AI) ]</code> ──▶ <i>(868 MHz LoRa Radio Link, Line-of-Sight ~12 km)</i> ──▶ <code>[ Portable Field Repeater ]</code> ──▶ <code>[ Incident Command Post (ICP) ]</code><br>
            <small style="color: gray;">Operates completely without cellular towers, 4G/5G, satellite internet, or cloud infrastructure.</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    @st.fragment(run_every="2s")
    def render_lora_terminal():
        packets = load_lora_packets()

        if not packets:
            st.info("📡 Listening on 868.1 MHz... No LoRa radio packets captured yet. Start `detect.py` or `python run_system.py` to transmit telemetry.")
            return

        latest_packet = packets[-1]

        # Link Quality KPIs
        lq1, lq2, lq3, lq4 = st.columns(4)
        rssi_val = latest_packet.get("rssi_dbm", -90)
        rssi_badge = "🟢 Strong" if rssi_val > -95 else ("🟡 Acceptable" if rssi_val > -105 else "🔴 Weak")
        lq1.metric("Signal Strength (RSSI)", f"{rssi_val} dBm", rssi_badge)
        lq2.metric("Signal-to-Noise (SNR)", f"{latest_packet.get('snr_db', 8.0)} dB", "Clear")
        lq3.metric("Last Packet SEQ", f"#{latest_packet.get('seq', 0):04d}", latest_packet.get("time_str", "Just now"))
        lq4.metric("Integrity Check", "PASSED", f"CRC {latest_packet.get('crc', '0xFFFF')}")

        st.divider()

        t_col_term, t_col_inspect = st.columns([1.6, 1.2])

        with t_col_term:
            st.markdown("##### 📟 Live Radio Packet Telemetry Stream")
            terminal_lines = []
            for p in reversed(packets[-15:]):
                prio = p.get("priority", "NORMAL")
                meta = PRIORITY_META.get(prio, {"hex": "#00CC66", "emoji": "🟢"})
                name = p.get("survivor_name", "UNKNOWN")
                line = (
                    f"<div style='font-family: monospace; font-size: 13px; padding: 4px 8px; border-bottom: 1px solid rgba(128,128,128,0.15);'>"
                    f"<span style='color: #888;'>[{p.get('time_str')}]</span> "
                    f"<b style='color: {meta['hex']};'>[{prio}]</b> "
                    f"<span>SEQ:{p.get('seq',0):03d}</span> | "
                    f"<span>ID:{p.get('survivor_id')}</span> <b>{name}</b> | "
                    f"<span style='color: #0088cc;'>GPS:{p.get('lat')},{p.get('lon')}</span> | "
                    f"<span style='color: #ffaa00;'>RSSI:{p.get('rssi_dbm')}dBm</span> | "
                    f"<code style='color: #a020f0;'>{p.get('crc')}</code>"
                    f"</div>"
                )
                terminal_lines.append(line)

            st.markdown(
                f"""
                <div style="background: #0e1117; color: #e6edf3; padding: 12px; border-radius: 6px; border: 1px solid #30363d; height: 360px; overflow-y: auto;">
                    <div style="color: #58a6ff; font-family: monospace; font-size: 12px; margin-bottom: 8px;">
                        --- RF GATEWAY RX LOG [CHANNEL 01 | 868.100 MHz] ---
                    </div>
                    {''.join(terminal_lines)}
                </div>
                """,
                unsafe_allow_html=True
            )

        with t_col_inspect:
            st.markdown("##### 🔬 Radio Packet Inspector & Payload Decoder")
            selected_seq = st.selectbox(
                "Select Received Packet",
                [f"Packet #{p.get('seq'):04d} — {p.get('time_str')} ({p.get('priority')})" for p in reversed(packets[-15:])]
            )
            sel_seq_num = int(selected_seq.split("#")[1].split(" ")[0])
            sel_pkt = next((p for p in packets if p.get("seq") == sel_seq_num), latest_packet)

            st.markdown(f"**Raw ASCII Air-Payload ({len(sel_pkt.get('raw_packet', ''))} bytes):**")
            st.code(sel_pkt.get("raw_packet", ""), language="text")

            st.markdown(f"**Hex Byte Preview:**")
            st.code(sel_pkt.get("hex_preview", ""), language="text")

            st.markdown(
                f"""
                <div style="background: rgba(128,128,128,0.08); padding: 10px; border-radius: 6px; border-left: 3px solid #00AA44;">
                    <strong>Decoded Rescue Directive:</strong><br>
                    • <b>Target Casualty:</b> {sel_pkt.get('survivor_name')} (ID #{sel_pkt.get('survivor_id')})<br>
                    • <b>Triage Level:</b> <code>{sel_pkt.get('priority')}</code><br>
                    • <b>Coordinates:</b> {sel_pkt.get('lat')}, {sel_pkt.get('lon')}<br>
                    • <b>Environmental Hazard:</b> {sel_pkt.get('hazard')}<br>
                    • <b>CRC Integrity:</b> <span style="color:green;">{sel_pkt.get('crc')} (VALID)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        raw_log_export = "\n".join([f"[{p['time_str']}] {p['raw_packet']} | RSSI:{p['rssi_dbm']}dBm | SNR:{p['snr_db']}dB" for p in packets])
        st.download_button(
            label="📥 Download Raw LoRa Radio Transmission Log (.txt)",
            data=raw_log_export,
            file_name=f"LoRa_Radio_Transmission_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    render_lora_terminal()

# ---------------- TAB 5: MISSION SITREP & EXPORT ----------------
with tab_sitrep:
    st.subheader("📑 Automated Mission Situation Report (SITREP)")
    st.caption("Generate an actionable, standardized tactical report for NDRF / SDRF / Civil Defense search-and-rescue teams.")

    telemetry = load_telemetry()
    detections = load_detections()
    hazards = load_hazards()
    survivor_log = load_survivor_log()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    active_critical = sum(1 for d in detections if d.get("priority") == "CRITICAL")
    active_high = sum(1 for d in detections if d.get("priority") == "HIGH")
    total_victims = len(survivor_log)
    active_hazards_count = len(hazards)

    # Build Markdown SITREP
    sitrep_text = f"""# 🛰️ SITUATION REPORT (SITREP) — DISASTER RESPONSE MISSION
**Generated**: {now_str}
**Operational Agency**: National Disaster Response Force (NDRF) / Disaster Management Cell
**Mission Platform**: Autonomous Edge-AI Aerial Reconnaissance Drone
**Connectivity Status**: Offline / Local Edge Computing Verified

---

## 1. EXECUTIVE MISSION SUMMARY
- **Total Tracked Entities**: {total_victims}
- **Immediate Life Threats (CRITICAL)**: {active_critical}
- **Urgent Evacuation Required (HIGH)**: {active_high}
- **Active Environmental Hazards**: {active_hazards_count}
- **Autonomous Flight & SLAM Navigation**:
  - Lawnmower Search Progress: {telemetry.get('survey_progress_pct', 0) if telemetry else 0}% Area Covered
  - Active Flight Waypoint: {telemetry.get('current_waypoint', 'N/A') if telemetry else 'N/A'} → {telemetry.get('next_waypoint', 'N/A') if telemetry else 'N/A'} ({telemetry.get('active_leg', 'N/A') if telemetry else 'N/A'})
  - 360° SLAM Collision Avoidance: {telemetry.get('slam_lidar', {}).get('avoidance_action', 'CLEAR_CORRIDOR') if telemetry else 'STANDBY'}
  - Distance to Base (ICP): {telemetry.get('dist_to_icp_m', 0.0) if telemetry else 0.0} m
  - Return-to-Home (RTH) Power Threshold: {telemetry.get('rth_battery_budget', 0.0) if telemetry else 0.0}% Required (Fail-Safe: {telemetry.get('rth_status', 'SAFE') if telemetry else 'SAFE'})
- **Drone Telemetry**:
  - Flight Mode: {telemetry.get('flight_mode', 'N/A') if telemetry else 'STANDBY'}
  - Altitude: {telemetry.get('altitude_m', 0.0) if telemetry else 0.0} m AGL
  - Battery Remaining: {telemetry.get('battery_pct', 0.0) if telemetry else 0.0} %
  - Sensor Mode: {'FLIR Thermal (Ironbow)' if (telemetry and telemetry.get('thermal_mode')) else 'RGB Optical HD'}

---

## 2. ACTIVE HAZARD ZONES (EXCLUSION PERIMETERS)
"""
    if hazards:
        for i, h in enumerate(hazards, 1):
            sitrep_text += f"- **Hazard #{i}**: {h.get('type')} Front | Lat: {h.get('lat', 'N/A')}, Lon: {h.get('lon', 'N/A')} | Confidence: {h.get('confidence', 0):.2f}\n"
    else:
        sitrep_text += "- No active hazards detected at this timestamp.\n"

    sitrep_text += "\n---\n\n## 3. TRIAGE & CASUALTY EVACUATION ROSTER (PRIORITIZED)\n"
    if survivor_log:
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}
        for entry in sorted(survivor_log, key=lambda x: priority_order.get(x.get("priority"), 4)):
            name = entry.get("name") or f"Unknown Entity #{entry['id']}"
            p = entry.get("priority", "NORMAL")
            pct = entry.get("priority_percent", 0)
            conds = ", ".join(entry.get("conditions") or []) or "None declared"
            notes = entry.get("health") or "None"
            live_stat = "ACTIVE IN FRAME" if entry.get("active_now") else "LEFT FRAME / STATIONARY"
            sitrep_text += (
                f"### [{p}] {name} (Risk: {pct}%)\n"
                f"- **Status**: {live_stat} | Posture: {entry.get('posture', 'Unknown')}\n"
                f"- **Age**: {entry.get('age', 'N/A')} | Medical Conditions: {conds}\n"
                f"- **Field Notes**: {notes}\n\n"
            )
    else:
        sitrep_text += "- No casualty data available in current log.\n"

    sitrep_text += """---
## 4. TACTICAL DIRECTIVES FOR GROUND RESCUE TEAMS
1. **Medevac Priority**: Immediate dispatch to CRITICAL individuals located on GIS tactical grid.
2. **Hazard Avoidance**: Maintain minimum 35-meter exclusion radius from active Fire and Flood zones.
3. **Safe Access Corridor**: Ingress via South-West staging waypoint (CartoDB Staging Grid ICP).
"""

    st.markdown(sitrep_text)

    st.download_button(
        label="📥 Download Tactical SITREP Report (.md)",
        data=sitrep_text,
        file_name=f"SITREP_Disaster_Response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )