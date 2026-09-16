# 🛰️ Autonomous Disaster Response Drone System — Edge-AI Triage & Command Center

> **Smart India Hackathon (SIH)**  
> **Domain**: Disaster Management, Autonomous Edge-AI & Search & Rescue (SAR) Robotics

---

## 📌 Problem Statement Alignment
During disasters (floods, fires, earthquakes, cyclones), responders need rapid situational awareness within the critical first golden hours. Ground-based assessments are slow and hazardous. This project implements an **autonomous drone edge-AI system** that operates in communication-constrained and GPS-denied/degraded environments with on-device intelligence:

* **On-Device Edge AI**: Runs lightweight YOLOv8 models (`yolov8n`, `fire_best`, `flood_best`, `yolov8n-pose`) and InsightFace locally without relying on cloud connectivity.
* **Autonomous Lawnmower Grid Planner & Search Optimization**: Generates and navigates a structured 6-leg parallel sweep survey grid (`WP-01` to `WP-06`) across disaster sectors to ensure 100% ground coverage with dynamic progress tracking.
* **360° LiDAR / SLAM Obstacle Avoidance Radar**: Multi-sector real-time rangefinder array (Front, Left, Right, Rear) detecting collapsed walls, dangling high-voltage cables, and debris with automated AI collision deflection vectors (`CLIMB`, `YAW SHIFT`).
* **GPS-Denied / Optical Flow VIO Navigation**: Maintains sub-meter position stability in indoor, underground, or collapsed structures using downward optical flow feature tracking (60 FPS) and 6-DOF IMU Extended Kalman Filtering (EKF).
* **Return-to-Home (RTH) Battery Fail-Safe**: Continuously computes return flight energy budget to Incident Command Post (ICP) datum and triggers fail-safe alerts before point of no return.
* **Multi-Sensor Payload Fusion**: Supports both **RGB Optical HD** and **Simulated FLIR Thermal (Ironbow)** sensor feeds to locate human heat signatures through smoke, foliage, or debris.
* **Autonomous Hazard Detection**: Real-time identification and localization of active fires, floodwaters, and environmental hazards.
* **Dynamic Medical Triage Matrix**: Integrates physical posture analysis (unconscious / lying down keypoints), visible trauma/blood heuristics, pre-registered vulnerabilities (wheelchair, asthma, blindness, cardiac issues), and proximity to hazards into a 0–100% priority score (`CRITICAL`, `HIGH`, `MODERATE`, `NORMAL`).
* **Tactical GIS Command Center**: Folium & WebGL PyDeck situational map displaying drone flight path, hazard exclusion buffer perimeters (35m radius), survivor markers, and **safe ground access evacuation corridors** for NDRF/SDRF ground rescue units.
* **Offline LoRa / Mesh Radio Telemetry**: In zero-connectivity or collapsed infrastructure zones, transmits emergency triage alerts over **868.1 MHz Sub-GHz LoRa radio packets** with dynamic RF signal metrics (RSSI, SNR, SF10) and CRC-16 validation.
* **Automated Mission SITREP**: One-click generation of official, timestamped National Disaster Response Force (NDRF) Situation Reports.

---

## ⚡ Quickstart: One-Click Launch

Run both the **Perception Engine** and **Tactical Command Center** simultaneously with a single command:

### Option 1: Via Python Launcher
```bash
# Using the local virtual environment (Webcam source 0)
python run_system.py

# Or specify a test video for indoor/hackathon demo
python run_system.py --source demo_flight.mp4
```

### Option 2: Windows Double-Click
Simply double-click **`start.bat`** in the project folder.

> 🌐 Once launched, the dashboard automatically opens in your default browser at: **`http://localhost:8501`**

---

## 🎮 Interactive Controls & Hotkeys

### In the OpenCV Drone Video Window:
* Press **`T`**: Toggle between **RGB Optical HD** and **FLIR Thermal (Ironbow)** sensor view.
* Press **`Q`**: Stop detection window (or press `Ctrl+C` in console to stop the entire system).

---

## 🏛️ System Architecture

```
                                [ Autonomous Drone Payload ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
              RGB Camera Feed                                FLIR Thermal Sensor
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                              [ On-Device Edge Perception Engine ]
                                        (detect.py)
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
              YOLOv8 Detection         YOLOv8 Keypoint          InsightFace +
               (Human / Fire /            Pose Model           Pre-Registered
                   Flood)               (Distress/Lying)          Embedding
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              ▼
                             [ Dynamic Triage Priority Engine ]
                               (Proximity, Posture, Conditions)
                                              │
                                              ▼
                             [ Atomic Real-Time Data Bus (JSON) ]
                          (telemetry.json / detections.json / hazards.json)
                                              │
                                              ▼
                           [ Tactical Ground Command Center ]
                               (registry_app.py - Streamlit)
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
               Live Folium GIS         Auto-Refreshing         Automated NDRF
                Tactical Map &          Priority Queue         SITREP Exporter
              Safe Access Route        & Mission Log
```

---

## 📊 Dynamic Triage Priority Scoring

| Factor | Points | Tactical Rationale |
| :--- | :---: | :--- |
| **Hazard Proximity** | **+50 pts** | Immediate physical danger (<250px from active fire/flood) |
| **Distress Posture** | **+30 pts** | Keypoint span ratio indicates lying down / unconscious |
| **Visible Injury / Trauma** | **+25 pts** | Blood/wound color signature |
| **Vulnerable Citizen Registry** | **+10 pts** | Pre-identified vulnerable individual |
| **High-Risk Age** | **+15 pts** | Children ($\le 10$) and Elderly ($\ge 60$) |
| **Disabilities / Medical Conditions** | **+10 to +30 pts** | Wheelchair (+30), Blindness (+25), Dementia (+25), Cardiac (+20), Asthma (+20) |
| **Asthma + Fire Synergistic Bonus**| **+15 pts** | High acute risk from smoke inhalation |

**Triage Priority Classification:**
* 🔴 **CRITICAL ($\ge 70$ pts)**: Immediate life threat — Air ambulance / medevac priority.
* 🟠 **HIGH ($\ge 40$ pts)**: Urgent ground extraction required.
* 🟡 **MODERATE ($\ge 20$ pts)**: Stable, non-life-threatening assistance needed.
* 🟢 **NORMAL ($< 20$ pts)**: Mobile, uninjured survivor.

---

## 📁 Repository Structure

```
SIH/
├── detect.py              # Edge-AI perception pipeline (YOLOv8 + Pose + Thermal + Face)
├── registry_app.py        # Streamlit Command Center (GIS, Auto-refresh, SITREP)
├── run_system.py          # Unified system launcher
├── start.bat              # One-click Windows runner
├── fire_best.pt           # Custom trained YOLOv8 Fire detection weights
├── flood_best.pt          # Custom trained YOLOv8 Flood detection weights
├── yolov8n.pt             # YOLOv8 nano human detection weights
├── yolov8n-pose.pt        # YOLOv8 keypoint pose estimation model
├── registry/
│   ├── registry.json      # Enrolled vulnerable citizen records
│   └── photos/            # Multi-angle facial training crops
├── detections.json        # Live active survivor feed
├── hazards.json           # Live active hazard coordinates
├── telemetry.json         # Drone telemetry, SLAM LiDAR rangefinder & survey grid
├── lora_packets.json      # Rolling 868.1 MHz Sub-GHz LoRa radio telemetry packets
├── survivor_log.json      # Persistent mission-wide survivor log
└── README.md              # Project documentation
```

---

## 👥 Team
Developed for the **Smart India Hackathon (SIH)**.
