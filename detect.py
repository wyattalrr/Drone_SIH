import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # prevents OpenMP crash (0xC0000409) when CUDA + multiple libs loaded
os.environ['OMP_NUM_THREADS'] = '1'            # stabilizes OpenMP threading on Windows
from ultralytics import YOLO
import cv2
import time
import math
import json
import os
import argparse
import numpy as np
import torch
import sys
import winsound
from insightface.app import FaceAnalysis

parser = argparse.ArgumentParser(description="Autonomous Disaster Response Drone Perception Engine")
parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or video file path)")
parser.add_argument("--no-gui", action="store_true", help="Run without graphical display window")
args = parser.parse_args()

def atomic_write_json(filepath, data):
    """Atomic write helper to prevent partial read race conditions with Streamlit."""
    tmp_path = f"{filepath}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        pass


def atomic_write_frame(filepath, frame_bgr):
    """Write an OpenCV BGR frame as JPEG for Streamlit live feed without NTFS rename locking."""
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with open(filepath, "wb") as f:
                f.write(buf.tobytes())
    except Exception:
        pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running YOLO models on: {DEVICE}")

person_model = YOLO("yolov8n.pt")
fire_model = YOLO("fire_best.pt")
flood_model = YOLO("flood_best.pt")
pose_model = YOLO("yolov8n-pose.pt")

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

cap = None
src = None
thermal_mode = False

LORA_PACKETS_FILE = "lora_packets.json"
LIVE_FRAME_FILE = "live_frame.jpg"   # shared annotated frame for Streamlit live feed
lora_transmission_history = []
lora_seq_id = 1
last_lora_broadcast_time = 0.0

# Autonomous Lawnmower Search Pattern Waypoints (6-leg SAR survey grid over disaster perimeter)
SAR_WAYPOINTS = [
    {"id": "WP-01", "lat": 26.9114, "lon": 75.7860, "name": "Ingress Point", "leg": "Leg 1: Ingress"},
    {"id": "WP-02", "lat": 26.9136, "lon": 75.7860, "name": "Sweep North A", "leg": "Leg 2: Northbound"},
    {"id": "WP-03", "lat": 26.9136, "lon": 75.7873, "name": "Cross East A", "leg": "Leg 3: Cross Track"},
    {"id": "WP-04", "lat": 26.9114, "lon": 75.7873, "name": "Sweep South B", "leg": "Leg 4: Southbound"},
    {"id": "WP-05", "lat": 26.9114, "lon": 75.7886, "name": "Cross East B", "leg": "Leg 5: Cross Track"},
    {"id": "WP-06", "lat": 26.9136, "lon": 75.7886, "name": "Sweep North C", "leg": "Leg 6: Egress"}
]

tracked_people = {}
next_id = 1

REAPPEAR_WINDOW = 120
REAPPEAR_MAX_DIST = 150   # a reappearing person's new position must be within this many pixels
                          # of where their ID was last seen — prevents a totally different,
                          # far-away person from inheriting someone else's locked-in name
MATCH_DIST = 45   # tightened from 60 — reduces ID-swap risk when people stand close together
REGISTRY_FILE = "registry/registry.json"
REGISTRY_CHECK_EVERY = 40
HAZARD_CHECK_EVERY = 5
FIRE_CONF = 0.6          # raised from 0.4 — yellow/warm room lights were false-triggering fire detection
FLOOD_CONF = 0.6   # raised from 0.4 — normal indoor/background scenes were false-triggering flood detection
POSTURE_CHECK_EVERY = 15
HAZARD_PROXIMITY_PX = 250
POSE_MATCH_RADIUS = 45   # tightened from 80 — reduces posture bleeding from a nearby different person
MATCH_THRESHOLD = 0.5   # cosine SIMILARITY (not distance) — higher = more similar. TUNE using the
                         # [DEBUG] similarity printouts below once you test with real people.
PERSON_CONF = 0.35       # lowered from 0.5 — improves detection on close-up webcam (partial body visible); raise to 0.5-0.6 if false positives appear

AGE_RISK_LOW = 10        # age <= this counts as higher-risk (child)
AGE_RISK_HIGH = 60       # age >= this counts as higher-risk (elderly)

# Weight = how much this condition impairs a person's ability to self-evacuate.
# Tune these numbers based on your team's judgment / mentor feedback.
CONDITION_WEIGHTS = {
    "Wheelchair-bound / Paralysis": 30,
    "Blind / Severe visual impairment": 25,
    "Cognitive impairment / Dementia": 25,
    "Cardiac condition": 20,
    "Respiratory condition (Asthma/COPD)": 20,
    "Pregnant": 20,
    "Deaf / Hearing impairment": 15,
    "Diabetic": 10,
    "Other / Unspecified": 5,
}
RESPIRATORY_FIRE_BONUS = 15   # extra points if a respiratory patient is specifically near FIRE (smoke risk)

# Realistic ceiling used to convert raw score into a 0-100% display (hazard 50 + posture 30 +
# injury 25 + registered base 10 + age 15 + ~2 strong conditions ~40 ≈ 170; capped a bit below that)
MAX_SCORE_FOR_PERCENT = 150

SEVERITY_POINTS = {"Low": 5, "Medium": 15, "High": 25}   # fallback only, for people registered before this change

INJURY_DETECTION_ENABLED = True   # set False instantly if it misfires during demo
INJURY_RED_RATIO_THRESHOLD = 0.12  # fraction of red pixels in crop to flag "possible injury" — TUNE by testing

SURVIVOR_LOG_FILE = "survivor_log.json"   # permanent record of EVERYONE ever detected this session, not just active


def load_registry_with_embeddings():
    if not os.path.exists(REGISTRY_FILE):
        return []
    with open(REGISTRY_FILE, "r") as f:
        raw = json.load(f)

    prepared = []
    for person in raw:
        photo_paths = person.get("photo_paths") or [person.get("photo_path")]
        embeddings = []
        for path in photo_paths:
            if not path:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                embeddings.append(faces[0].embedding)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            prepared.append({
                "name": person["name"],
                "health": person.get("health"),
                "age": person.get("age"),
                "conditions": person.get("conditions", []),   # new — specific condition list
                "severity": person.get("severity"),            # legacy fallback (old Low/Medium/High entries)
                "embedding": avg_embedding
            })
            print(f"Registry loaded: {person['name']} ({len(embeddings)} photo(s) averaged)")
        else:
            print(f"Could not process any registry photo for {person['name']}")
    return prepared


print("Loading registry (one-time)...")
registry = load_registry_with_embeddings()
print(f"Registry ready: {len(registry)} people loaded.")


def match_registry(face_crop):
    if face_crop.size == 0 or not registry:
        return None
    faces = face_app.get(face_crop)
    if not faces:
        return None

    # If more than one face shows up in this crop (bounding boxes overlapping
    # because two people are standing close together), pick whichever face is
    # closest to the CENTER of this crop — that's almost always the actual
    # person this box belongs to, not someone standing next to them.
    if len(faces) > 1:
        crop_h, crop_w = face_crop.shape[:2]
        cx, cy = crop_w / 2, crop_h / 2

        def dist_from_crop_center(f):
            fx1, fy1, fx2, fy2 = f.bbox
            fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            return math.hypot(fcx - cx, fcy - cy)

        target_face = min(faces, key=dist_from_crop_center)
    else:
        target_face = faces[0]

    embedding = target_face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    best_match, best_sim = None, MATCH_THRESHOLD
    for person in registry:
        sim = float(np.dot(embedding, person["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best_match = person
    if best_match:
        print(f"[RECOG] Matched registered citizen: {best_match['name']} (score: {best_sim:.2f})")
    return best_match


def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def box_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def detect_visible_injury(crop):
    """
    HEURISTIC ONLY — not a trained injury/blood classifier. Flags a crop
    as "possible injury" if a large fraction of it is red/blood-colored
    (HSV red range). This WILL false-positive on red clothing, red
    lighting, etc. Treat the output as a supporting signal to raise
    attention, never as a confirmed diagnosis. Tune
    INJURY_RED_RATIO_THRESHOLD against real test footage before demo day,
    and use INJURY_DETECTION_ENABLED to switch it off instantly if it
    misbehaves live.
    """
    if crop.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = float(np.count_nonzero(mask)) / (crop.shape[0] * crop.shape[1])
    return red_ratio > INJURY_RED_RATIO_THRESHOLD, red_ratio


def classify_posture(keypoints):
    """
    Only judges posture when BOTH an upper-body point (shoulder) AND a
    lower-body point (hip/ankle) are visible. On a close-up webcam feed
    (face/shoulders only, no lower body in frame) this now returns
    "Unknown" instead of incorrectly guessing "Lying down" — the old
    version misfired here because two shoulder points alone always have
    a small vertical spread and a larger horizontal spread, which used
    to trigger a false "Lying down" result every time.
    """
    pts = {}
    for idx in [5, 6, 11, 12, 15, 16]:
        x, y, conf = keypoints[idx]
        if conf > 0.3:
            pts[idx] = (x, y)

    shoulder_visible = (5 in pts) or (6 in pts)
    lower_body_visible = any(idx in pts for idx in [11, 12, 15, 16])

    if not (shoulder_visible and lower_body_visible):
        return "Unknown"

    all_pts = list(pts.values())
    ys = [p[1] for p in all_pts]
    xs = [p[0] for p in all_pts]
    vertical_span = max(ys) - min(ys)
    horizontal_span = max(xs) - min(xs)
    if vertical_span < horizontal_span * 0.6:
        return "Lying down / Distress"
    return "Upright"


def assign_id(center, tracked_people, claimed_ids):
    global next_id
    now = time.time()

    best_id, best_dist = None, MATCH_DIST
    for pid, data in tracked_people.items():
        if data["active"] and pid not in claimed_ids:
            dist = math.hypot(center[0]-data["center"][0], center[1]-data["center"][1])
            if dist < best_dist:
                best_dist, best_id = dist, pid
    if best_id is not None:
        return best_id

    best_id, best_time = None, None
    for pid, data in tracked_people.items():
        if not data["active"] and pid not in claimed_ids and (now - data["last_seen"]) < REAPPEAR_WINDOW:
            dist = math.hypot(center[0]-data["center"][0], center[1]-data["center"][1])
            if dist < REAPPEAR_MAX_DIST:   # must ALSO be near where this ID was last seen
                if best_time is None or data["last_seen"] > best_time:
                    best_time, best_id = data["last_seen"], pid
    if best_id is not None:
        return best_id

    new_id = next_id
    tracked_people[new_id] = {
        "center": center, "first_seen": now, "last_seen": now,
        "start_center": center, "active": True,
        "name": None, "health": None, "age": None, "conditions": [], "severity": None,
        "priority": "NORMAL", "posture": "Unknown", "status": "STATIONARY",
        "injury": False, "frames_since_check": 0
    }
    next_id += 1
    return new_id


def initialize_capture(source):
    """
    Robustly initialize video capture.
    Tries multiple backends (CAP_DSHOW, standard) and tests reading frames.
    If a requested camera index fails (e.g. index 0), automatically probes fallback
    camera indices (e.g. index 1) to ensure the perception engine finds a working camera device.
    """
    is_cam_idx = False
    try:
        req_idx = int(source)
        is_cam_idx = True
    except (ValueError, TypeError):
        req_idx = source

    if not is_cam_idx:
        print(f"[*] Opening video source / file: {req_idx}", flush=True)
        c = cv2.VideoCapture(req_idx)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return c, req_idx
        print(f"[ERROR] Failed to open video file/stream: {req_idx}", flush=True)
        return None, req_idx

    # Candidate camera indices: requested index first, then common alternatives
    candidates = [req_idx]
    for alt in [1, 0, 2, 3]:
        if alt not in candidates:
            candidates.append(alt)

    for idx in candidates:
        is_requested = (idx == req_idx)
        prefix = f"[*] Connecting to camera index {idx}" if is_requested else f"[*] Probing fallback camera index {idx}"
        print(f"{prefix}...", flush=True)

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"
            try:
                c = cv2.VideoCapture(idx, backend)
                if not c.isOpened():
                    c.release()
                    continue
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[*] Camera {idx} opened via {backend_name} — validating live video stream...", flush=True)
                time.sleep(0.6)   # brief settle time for hardware auto-exposure
                got_frame = False
                clean_count = 0
                for _ in range(15):
                    ret, test_frame = c.read()
                    if ret and test_frame is not None:
                        # Sanity-check: reject pure-noise / all-black frames
                        mean_val = float(np.mean(test_frame))
                        std_val = float(np.std(test_frame))
                        # Must have real light (mean > 4.0) and spatial contrast (std > 4.0)
                        if mean_val > 4.0 and 4.0 < std_val < 95.0:
                            clean_count += 1
                            if clean_count >= 2:
                                got_frame = True
                                break
                        else:
                            clean_count = 0
                    time.sleep(0.1)
                if got_frame:
                    print(f"[OK] Camera {idx} streaming clean frames via {backend_name} (640x480).", flush=True)
                    return c, idx
                else:
                    print(f"[!] Camera {idx} ({backend_name}) dark or unreadable — trying next...", flush=True)
                    c.release()
            except Exception as e:
                print(f"[!] Error trying camera {idx} ({backend_name}): {e}", flush=True)

    return None, req_idx

print("\n" + "=" * 70, flush=True)
print("[*] Initializing camera perception sensor payload...", flush=True)
cap, src = initialize_capture(args.source)

if cap is None:
    print("!" * 70, flush=True)
    print(f"[FATAL] Could not initialize camera device (requested: {args.source}).", flush=True)
    print("Troubleshooting steps:", flush=True)
    print("  1. Verify your USB camera (QPC-1015) is connected securely.", flush=True)
    print("  2. Check if another app (Windows Camera, browser, Zoom) is holding the feed.", flush=True)
    print("  3. Check Windows Settings -> Privacy & Security -> Camera -> Allow desktop apps.", flush=True)
    print("  4. Check the physical privacy shutter / lens cover on your camera.", flush=True)
    print("!" * 70 + "\n", flush=True)
    offline_telemetry = {
        "flight_mode": "STANDBY / LANDED",
        "status": "CAMERA_OFFLINE",
        "timestamp": time.time(),
        "battery_pct": 0.0,
        "altitude_m": 0.0,
        "speed_mps": 0.0,
        "heading_deg": 0,
        "active_survivors": 0,
        "active_hazards": 0,
        "thermal_mode": False
    }
    atomic_write_json("telemetry.json", offline_telemetry)
    sys.exit(1)

# Notify telemetry immediately that camera is active
drone_init_telemetry = {
    "flight_mode": "AUTO_LAWNMOWER_SURVEY",
    "status": "CAMERA_ONLINE",
    "timestamp": time.time(),
    "active_survivors": 0,
    "active_hazards": 0,
    "thermal_mode": False,
    "current_waypoint": SAR_WAYPOINTS[0]["id"],
    "next_waypoint": SAR_WAYPOINTS[1]["id"],
    "active_leg": SAR_WAYPOINTS[0]["leg"],
    "search_grid_waypoints": SAR_WAYPOINTS
}
atomic_write_json("telemetry.json", drone_init_telemetry)
print("[OK] Drone camera online. Perception pipeline active.\n" + "=" * 70 + "\n", flush=True)

last_fire_boxes = []
last_flood_boxes = []
last_pose_data = []   # list of (center, posture)
frame_num = 0
fps_frame_count = 0
fps_timer = time.time()
consecutive_read_failures = 0
MAX_CONSECUTIVE_FAILURES = 30   # ~2-3 seconds of failures before we treat it as a real disconnect
LIVE_FRAME_WRITE_EVERY = 2    # write shared JPEG every N frames (keeps Streamlit feed ~15-20 FPS without I/O overload)

while True:
    ret, frame = cap.read()
    if not ret:
        if not isinstance(src, int) and os.path.exists(str(src)):
            # Video file loop for continuous hackathon demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        consecutive_read_failures += 1
        print(f"Webcam frame grab failed ({consecutive_read_failures}/{MAX_CONSECUTIVE_FAILURES}) — retrying...", flush=True)
        if consecutive_read_failures == 15 and isinstance(src, int):
            print("[*] Attempting soft camera reconnection...", flush=True)
            try:
                cap.release()
                cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception:
                pass
        if consecutive_read_failures >= MAX_CONSECUTIVE_FAILURES:
            print("Webcam appears disconnected. Stopping.", flush=True)
            break
        cv2.waitKey(100)
        continue
    consecutive_read_failures = 0

    frame_num += 1
    person_results = person_model(frame, conf=PERSON_CONF, classes=[0], imgsz=320, verbose=False, device=DEVICE)

    if frame_num % HAZARD_CHECK_EVERY == 0:
        fire_results = fire_model(frame, conf=FIRE_CONF, imgsz=320, verbose=False, device=DEVICE)
        flood_results = flood_model(frame, conf=FLOOD_CONF, imgsz=320, verbose=False, device=DEVICE)
        last_fire_boxes = fire_results[0].boxes
        last_flood_boxes = flood_results[0].boxes

    if frame_num % POSTURE_CHECK_EVERY == 0:
        pose_results = pose_model(frame, conf=0.5, imgsz=320, verbose=False, device=DEVICE)
        new_pose_data = []
        p_boxes = pose_results[0].boxes
        p_keypoints = pose_results[0].keypoints
        for i in range(len(p_boxes)):
            px1, py1, px2, py2 = map(int, p_boxes[i].xyxy[0])
            kpts = p_keypoints.data[i].cpu().numpy() if p_keypoints is not None else None
            posture = classify_posture(kpts) if kpts is not None else "Unknown"
            new_pose_data.append((box_center(px1, py1, px2, py2), posture))
        last_pose_data = new_pose_data

    annotated = frame.copy()
    now = time.time()

    for pid in tracked_people:
        tracked_people[pid]["active"] = False

    already_checked_this_frame = False
    claimed_ids = set()   # prevents two different detections THIS frame from sharing one tracked ID

    hazard_boxes = []
    for box in last_fire_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        hazard_boxes.append((x1, y1, x2, y2, "Fire"))
    for box in last_flood_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        hazard_boxes.append((x1, y1, x2, y2, "Flood"))

    for box in person_results[0].boxes:
        center = get_center(box)
        pid = assign_id(center, tracked_people, claimed_ids)
        claimed_ids.add(pid)

        data = tracked_people[pid]
        data["center"] = center
        data["last_seen"] = now
        data["active"] = True
        if "frames_since_check" not in data:
            data["frames_since_check"] = 0

        duration = now - data["first_seen"]
        move_dist = math.hypot(center[0]-data["start_center"][0], center[1]-data["start_center"][1])
        status = "STATIONARY" if move_dist < 40 else "MOVING"
        data["status"] = status

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        data["frames_since_check"] += 1
        # Efficiency optimization: Prioritize checking unidentified people to avoid frame lag spikes
        needs_check = (data.get("name") is None and data["frames_since_check"] >= REGISTRY_CHECK_EVERY) or (data["frames_since_check"] >= 150)
        if not already_checked_this_frame and needs_check:
            data["frames_since_check"] = 0
            already_checked_this_frame = True
            match = match_registry(crop)
            if match is not None:
                data["name"] = match["name"]
                data["health"] = match["health"]
                data["age"] = match.get("age")
                data["conditions"] = match.get("conditions", [])
                data["severity"] = match.get("severity")
                data["match_miss_count"] = 0
            elif data.get("name") is not None:
                # Debounce: only clear after 3 consecutive misses if already identified
                data["match_miss_count"] = data.get("match_miss_count", 0) + 1
                if data["match_miss_count"] >= 3:
                    data["name"] = None
                    data["health"] = None
                    data["age"] = None
                    data["conditions"] = []
                    data["severity"] = None
                    data["match_miss_count"] = 0

        # --- match nearest pose reading to this tracked person ---
        posture = "Unknown"
        best_pdist = POSE_MATCH_RADIUS
        for p_center, p_posture in last_pose_data:
            pdist = math.hypot(center[0]-p_center[0], center[1]-p_center[1])
            if pdist < best_pdist:
                best_pdist = pdist
                posture = p_posture
        data["posture"] = posture

        # --- hazard proximity ---
        near_hazard, min_hdist = None, None
        for hb in hazard_boxes:
            hc = box_center(*hb[:4])
            hdist = math.hypot(center[0]-hc[0], center[1]-hc[1])
            if min_hdist is None or hdist < min_hdist:
                min_hdist, near_hazard = hdist, hb[4]
        hazard_nearby = near_hazard is not None and min_hdist < HAZARD_PROXIMITY_PX

        # --- visible injury heuristic (see docstring on detect_visible_injury) ---
        injury_detected = False
        if INJURY_DETECTION_ENABLED:
            injury_detected, red_ratio = detect_visible_injury(crop)
        data["injury"] = injury_detected

        # --- combined live priority score ---
        score = 0
        if hazard_nearby:
            score += 50
        if posture == "Lying down / Distress":
            score += 30
        if injury_detected:
            score += 25
        if data["name"] is not None:
            score += 10   # base bonus just for being a known vulnerable person
            age = data.get("age")
            if age is not None and (age <= AGE_RISK_LOW or age >= AGE_RISK_HIGH):
                score += 15

            conditions = data.get("conditions") or []
            if conditions:
                for cond in conditions:
                    score += CONDITION_WEIGHTS.get(cond, 5)
                if "Respiratory condition (Asthma/COPD)" in conditions and near_hazard == "Fire":
                    score += RESPIRATORY_FIRE_BONUS
            elif data.get("severity"):
                # legacy entries registered before condition-list existed
                score += SEVERITY_POINTS.get(data["severity"], 5)

        if score >= 70:
            priority = "CRITICAL"
        elif score >= 40:
            priority = "HIGH"
        elif score >= 20:
            priority = "MODERATE"
        else:
            priority = "NORMAL"

        if priority == "CRITICAL" and data.get("priority") != "CRITICAL":
            try:
                winsound.Beep(1200, 400)   # only fires on the transition INTO critical, not every frame
            except Exception:
                pass   # non-Windows or no sound device — don't let this crash the loop
        data["priority"] = priority

        # normalize raw score into a 0-100% "risk level" for finer-grained display
        # alongside the category (e.g. two CRITICAL people aren't equally critical)
        priority_percent = min(100, round((score / MAX_SCORE_FOR_PERCENT) * 100))
        data["priority_percent"] = priority_percent

        label_name = data["name"] if data["name"] else "Unknown Entity"
        if priority == "CRITICAL":
            box_color = (255, 0, 255)   # magenta — kept distinct from Fire (red) and Flood (blue)
        elif priority == "HIGH":
            box_color = (0, 165, 255)
        elif priority == "MODERATE":
            box_color = (0, 255, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
        label_y = y1 - 12 if y1 - 12 > 20 else y1 + 25   # never draw text off the top edge
        cv2.putText(annotated, label_name,
                    (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    for box in last_fire_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = fire_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_y = y1 - 10 if y1 - 10 > 15 else y1 + 20
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for box in last_flood_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = flood_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label_y = y1 - 10 if y1 - 10 > 15 else y1 + 20
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    hazard_log = []
    for i, box in enumerate(last_fire_boxes):
        hazard_log.append({
            "type": "Fire", "confidence": float(box.conf[0]),
            "lat": 26.9124 + (i * 0.0004) + 0.0006,
            "lon": 75.7873 + (i * 0.0004) + 0.0006
        })
    for i, box in enumerate(last_flood_boxes):
        hazard_log.append({
            "type": "Flood", "confidence": float(box.conf[0]),
            "lat": 26.9124 + (i * 0.0004) - 0.0006,
            "lon": 75.7873 + (i * 0.0004) - 0.0006
        })
    atomic_write_json("hazards.json", hazard_log)

    live_data = []
    for pid, data in tracked_people.items():
        if data["active"]:
            live_data.append({
                "id": pid,
                "name": data["name"],
                "type": "Survivor",
                "status": data.get("status", "STATIONARY"),
                "posture": data.get("posture", "Unknown"),
                "priority": data.get("priority", "NORMAL"),
                "priority_percent": data.get("priority_percent", 0),
                "lat": 26.9124 + (pid * 0.0003),
                "lon": 75.7873 + (pid * 0.0003)
            })
    atomic_write_json("detections.json", live_data)

    survivor_log = []
    for pid, data in tracked_people.items():
        survivor_log.append({
            "id": pid,
            "name": data["name"],
            "priority": data.get("priority", "NORMAL"),
            "priority_percent": data.get("priority_percent", 0),
            "posture": data.get("posture", "Unknown"),
            "status": data.get("status", "STATIONARY"),
            "age": data.get("age"),
            "conditions": data.get("conditions", []),
            "health": data.get("health"),
            "active_now": data["active"],
            "first_seen": data["first_seen"],
            "last_seen": data["last_seen"]
        })
    atomic_write_json(SURVIVOR_LOG_FILE, survivor_log)

    # --- Autonomous Lawnmower Search Navigation & SLAM Telemetry Engine ---
    total_wp = len(SAR_WAYPOINTS)
    wp_cycle_duration = 180  # ~6 seconds per survey leg
    total_mission_frames = total_wp * wp_cycle_duration

    current_leg_idx = int((frame_num // wp_cycle_duration) % total_wp)
    next_leg_idx = (current_leg_idx + 1) % total_wp
    t_leg = (frame_num % wp_cycle_duration) / float(wp_cycle_duration)

    wp_start = SAR_WAYPOINTS[current_leg_idx]
    wp_end = SAR_WAYPOINTS[next_leg_idx]

    cur_lat = wp_start["lat"] + t_leg * (wp_end["lat"] - wp_start["lat"])
    cur_lon = wp_start["lon"] + t_leg * (wp_end["lon"] - wp_start["lon"])

    survey_progress_pct = min(100, int(((frame_num % (total_mission_frames * 2)) / float(total_mission_frames * 2)) * 100))

    dist_to_icp_m = math.hypot((cur_lat - 26.9108) * 111000, (cur_lon - 75.7858) * 111000)
    rth_battery_budget = round(max(12.0, (dist_to_icp_m / 100.0) * 1.8 + 8.0), 1)
    battery_remaining = max(15.0, round(99.0 - (frame_num * 0.012), 1))
    rth_status = "CRITICAL_RTH_TRIGGERED" if battery_remaining <= rth_battery_budget else "SAFE_NOMINAL"

    # Simulated 360-Degree LiDAR / SLAM Distance Sensors (Front, Left, Right, Rear)
    d_front = round(max(1.2, 4.6 + 2.0 * math.sin(frame_num * 0.08) - (1.6 if last_fire_boxes else 0)), 1)
    d_left = round(max(0.8, 3.4 + 1.6 * math.cos(frame_num * 0.07)), 1)
    d_right = round(max(1.5, 5.0 + 2.1 * math.sin(frame_num * 0.06)), 1)
    d_rear = round(max(2.0, 5.8 + 1.4 * math.cos(frame_num * 0.05)), 1)

    slam_obstacles = []
    avoidance_action = "CLEAR_CORRIDOR"
    if d_front < 2.0:
        slam_obstacles.append({"sector": "FRONT", "dist_m": d_front, "hazard": "COLLAPSED_CONCRETE_WALL", "severity": "WARNING"})
        avoidance_action = "YAW +15° RIGHT | CLIMB +2.0m"
    if d_left < 1.0:
        slam_obstacles.append({"sector": "LEFT", "dist_m": d_left, "hazard": "EXPOSED_POWER_CABLES", "severity": "CRITICAL"})
        avoidance_action = "YAW +20° RIGHT | LATERAL SHIFT"
    elif d_right < 1.4:
        slam_obstacles.append({"sector": "RIGHT", "dist_m": d_right, "hazard": "DEBRIS_TREE_CANOPY", "severity": "WARNING"})
        avoidance_action = "YAW -15° LEFT"

    drone_telemetry = {
        "lat": round(cur_lat, 6),
        "lon": round(cur_lon, 6),
        "altitude_m": round(24.0 + 1.5 * math.sin(frame_num * 0.03), 1),
        "speed_mps": round(4.2 + 0.3 * math.cos(frame_num * 0.04), 1),
        "battery_pct": battery_remaining,
        "heading_deg": int(math.degrees(math.atan2(wp_end["lon"] - wp_start["lon"], wp_end["lat"] - wp_start["lat"])) % 360),
        "flight_mode": "AUTO_LAWNMOWER_SURVEY",
        "thermal_mode": thermal_mode,
        "timestamp": time.time(),
        "active_survivors": sum(1 for d in tracked_people.values() if d["active"]),
        "active_hazards": len(last_fire_boxes) + len(last_flood_boxes),
        # Autonomous Navigation & SLAM Suite fields:
        "current_waypoint": wp_start["id"],
        "next_waypoint": wp_end["id"],
        "active_leg": wp_start["leg"],
        "survey_progress_pct": survey_progress_pct,
        "dist_to_icp_m": round(dist_to_icp_m, 1),
        "rth_battery_budget": rth_battery_budget,
        "rth_status": rth_status,
        "nav_engine": "GPS_LOCK_RTK",
        "satellites": 14,
        "slam_lidar": {
            "front_m": d_front,
            "left_m": d_left,
            "right_m": d_right,
            "rear_m": d_rear,
            "obstacles": slam_obstacles,
            "avoidance_action": avoidance_action
        },
        "search_grid_waypoints": SAR_WAYPOINTS
    }
    atomic_write_json("telemetry.json", drone_telemetry)

    # Simulated LoRa Mesh Radio Transmission (868.1 MHz Sub-GHz Offline Telemetry)
    if (now - last_lora_broadcast_time) >= 2.5:
        last_lora_broadcast_time = now
        active_tracked = [d for d in live_data if d.get("priority") in ["CRITICAL", "HIGH", "MODERATE"]]
        target_entity = active_tracked[0] if active_tracked else (live_data[0] if live_data else None)

        dist_km = math.hypot(drone_telemetry["lat"] - 26.9108, drone_telemetry["lon"] - 75.7858) * 111.0
        simulated_rssi = max(-115, round(-78 - (16 * math.log10(max(0.1, dist_km + 0.08))) - (np.random.rand() * 3), 1))
        simulated_snr = round(max(-4.0, min(12.0, 9.5 - (dist_km * 2.2) + (np.random.rand() * 2))), 1)

        prio_tag = target_entity["priority"] if target_entity else "BEACON_IDLE"
        surv_id = target_entity["id"] if target_entity else 0
        surv_name = (target_entity["name"] if target_entity and target_entity["name"] else "UNKNOWN").upper().replace(" ", "_")
        lat_str = f"{target_entity['lat']:.5f}" if target_entity else f"{drone_telemetry['lat']:.5f}"
        lon_str = f"{target_entity['lon']:.5f}" if target_entity else f"{drone_telemetry['lon']:.5f}"
        haz_tag = hazard_log[0]["type"].upper() if hazard_log else "CLEAR"
        post_tag = target_entity["posture"].upper().replace(" ", "_") if target_entity else "N/A"

        raw_payload = f"!LORA_SOS|SEQ:{lora_seq_id:04d}|SRC:DRONE_ALPHA|PRIO:{prio_tag}|ID:{surv_id}|NAME:{surv_name}|GPS:{lat_str},{lon_str}|HAZ:{haz_tag}|POST:{post_tag}"
        crc_val = sum(ord(c) for c in raw_payload) & 0xFFFF
        formatted_packet = f"{raw_payload}|CRC:0x{crc_val:04X}"
        hex_payload = " ".join(f"{ord(c):02X}" for c in formatted_packet[:32]) + " ..."

        packet_entry = {
            "seq": lora_seq_id,
            "timestamp": now,
            "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
            "frequency_mhz": 868.1,
            "spreading_factor": "SF10",
            "bandwidth_khz": 125,
            "rssi_dbm": simulated_rssi,
            "snr_db": simulated_snr,
            "priority": prio_tag,
            "survivor_name": surv_name if target_entity else "N/A",
            "survivor_id": surv_id,
            "lat": float(lat_str),
            "lon": float(lon_str),
            "hazard": haz_tag,
            "posture": target_entity.get("posture", "N/A") if target_entity else "N/A",
            "raw_packet": formatted_packet,
            "hex_preview": hex_payload,
            "crc": f"0x{crc_val:04X}",
            "crc_valid": True
        }
        lora_seq_id += 1
        lora_transmission_history.append(packet_entry)
        if len(lora_transmission_history) > 60:
            lora_transmission_history.pop(0)

        atomic_write_json(LORA_PACKETS_FILE, lora_transmission_history)

    fps_frame_count += 1
    if fps_frame_count >= 30:
        elapsed = time.time() - fps_timer
        fps = fps_frame_count / elapsed if elapsed > 0 else 0
        active_count = sum(1 for d in tracked_people.values() if d["active"])
        sensor_str = "FLIR THERMAL" if thermal_mode else "RGB"
        print(f"FPS: {fps:.1f} | People in frame: {active_count} | Sensor: {sensor_str}", flush=True)
        fps_frame_count = 0
        fps_timer = time.time()

    # Build display/export frame (shared by OpenCV window AND Streamlit live feed)
    if thermal_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_overlay = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        display_frame = cv2.addWeighted(annotated, 0.60, thermal_overlay, 0.40, 0)
        cv2.rectangle(display_frame, (10, 10), (330, 68), (0, 0, 0), -1)
        cv2.putText(display_frame, "[THERMAL FLIR PAYLOAD]", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.putText(display_frame, "Mode: IRONBOW | Press 'T': RGB", (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        display_frame = annotated.copy()
        cv2.rectangle(display_frame, (10, 10), (330, 68), (0, 0, 0), -1)
        cv2.putText(display_frame, "[RGB SENSOR PAYLOAD]", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'T': Thermal FLIR | 'Q': Quit", (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Write shared JPEG frame for Streamlit live feed (every N frames to keep I/O light, plus immediate on frame 1)
    if frame_num == 1 or frame_num % LIVE_FRAME_WRITE_EVERY == 0:
        atomic_write_frame(LIVE_FRAME_FILE, display_frame)

    if not args.no_gui:
        cv2.imshow("Disaster Response - Autonomous Drone Edge-AI", display_frame)

        # Check if window was closed via 'X' button (only after initial frames to avoid false positives during window creation)
        if frame_num > 5:
            try:
                prop = cv2.getWindowProperty("Disaster Response - Autonomous Drone Edge-AI", cv2.WND_PROP_AUTOSIZE)
                if prop < 0:
                    print("[*] Camera window closed by user.", flush=True)
                    break
            except cv2.error:
                print("[*] Camera window closed by user.", flush=True)
                break
            except Exception:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[*] User requested perception engine shutdown ('Q').", flush=True)
            break
        elif key == ord('t'):
            thermal_mode = not thermal_mode
            print(f"Sensor payload switched: {'THERMAL FLIR' if thermal_mode else 'RGB'}", flush=True)
    else:
        # Headless mode: still poll for no-op delay to keep loop rate reasonable
        cv2.waitKey(1)

cap.release()
if not args.no_gui:
    cv2.destroyAllWindows()

# Inform Command Center that camera feed has ended while keeping mission data intact
try:
    offline_telemetry = {
        "flight_mode": "STANDBY / LANDED",
        "status": "CAMERA_OFFLINE",
        "timestamp": time.time(),
        "battery_pct": 0.0,
        "altitude_m": 0.0,
        "speed_mps": 0.0,
        "heading_deg": 0,
        "active_survivors": 0,
        "active_hazards": 0,
        "thermal_mode": False
    }
    atomic_write_json("telemetry.json", offline_telemetry)
    print("[*] Drone telemetry status set to STANDBY / LANDED.")
except Exception:
    pass