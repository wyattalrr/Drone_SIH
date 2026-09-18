#!/usr/bin/env python3
"""
=============================================================================
Autonomous Disaster Response Drone System — Unified Launcher
Smart India Hackathon (SIH)
=============================================================================
Launches both the Edge-AI Perception Engine (detect.py) and the
Tactical Command Center Dashboard (registry_app.py) simultaneously.
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
import webbrowser
import argparse

def get_python_executable():
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sih_venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def print_banner():
    banner = r"""
=============================================================================
   __  ___      ____  _                     __            ___   ____
  /  |/  /_  __/ / /_(_)   __  ______ _____/ /___  _____ /   | /  _/
 / /|_/ / / / / / __/ /   / / / / __ `/ __  / __ \/ ___// /| | / /  
/ /  / / /_/ / / /_/ /   / /_/ / /_/ / /_/ / /_/ / /   / ___ |_/ /   
/_/  /_/\__,_/_/\__/_/    \__, /\__,_/\__,_/\____/_/   /_/  |_/___/   
                         /____/                                      
   AUTONOMOUS DISASTER RESPONSE DRONE SYSTEM | SIH MISSION RUNNER
=============================================================================
- Multi-Hazard AI: Fire, Flood, Human Detection & Keypoint Pose Distress
- Multi-Sensor: RGB Optical HD + Simulated FLIR Ironbow Thermal Camera
- Real-Time GIS: Dynamic Triage, Safe Ground Evacuation Corridor, SITREP
=============================================================================
    """
    print(banner)

def get_available_cameras():
    """
    Detect physically connected cameras using Windows PnP.
    Returns a list of dicts: [{'index': 0, 'name': '...', 'is_external': bool}]
    """
    cameras = []
    if sys.platform == "win32":
        try:
            cmd = "Get-PnpDevice -Class Camera -PresentOnly | Select-Object FriendlyName | ConvertTo-Json"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                raw = json.loads(res.stdout)
                if isinstance(raw, dict):
                    raw = [raw]
                for idx, d in enumerate(raw):
                    name = d.get("FriendlyName", f"Camera {idx}").strip()
                    is_ext = any(k in name.lower() for k in ["usb", "qpc", "webcam", "external", "5mp", "drone"])
                    cameras.append({
                        "index": idx,
                        "name": name,
                        "is_external": is_ext
                    })
        except Exception:
            pass

    if not cameras:
        cameras = [{"index": 0, "name": "HP Wide Vision HD Camera (Built-in)", "is_external": False}]

    return cameras

def kill_stale_processes():
    """Kill any leftover detect.py or streamlit processes from previous runs before starting fresh.
    This prevents multiple OpenCV windows and camera contention causing black frames."""
    try:
        import psutil
        my_pid = os.getpid()
        killed = False
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if p.info['pid'] == my_pid:
                    continue
                cmdline = " ".join(p.info['cmdline'] or []).lower()
                if "detect.py" in cmdline or ("streamlit" in cmdline and "registry_app.py" in cmdline):
                    p.terminate()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if killed:
            print("[*] Cleared stale background drone/command-center processes.", flush=True)
            time.sleep(1.5)
    except Exception:
        pass

def check_and_clear_port(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_in_use = (s.connect_ex(('127.0.0.1', port)) == 0)
    if is_in_use:
        print(f"[!] Port {port} is still in use — waiting for release...", flush=True)
        time.sleep(2.0)

def main():
    parser = argparse.ArgumentParser(description="Launch the Autonomous Disaster Response Drone System")
    parser.add_argument("--source", type=str, default=None, help="Video source (0, 1 for webcam, or path to test video)")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit dashboard port (default: 8501)")
    parser.add_argument("--no-gui", action="store_true", help="Run detection headless without OpenCV window")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    kill_stale_processes()
    print_banner()

    python_exe = get_python_executable()
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Hardware Camera Discovery
    cameras = get_available_cameras()
    selected_source = args.source
    selected_camera_name = "User specified video source"

    if selected_source is None:
        print("[*] Probing connected camera hardware...")
        external_cam = next((c for c in cameras if c.get("is_external")), None)
        chosen_cam = external_cam or (cameras[0] if cameras else None)

        for c in cameras:
            tag = "(External USB Camera)" if c.get("is_external") else "(Built-in Camera)"
            is_chosen = (chosen_cam and c["index"] == chosen_cam["index"])
            mark = " -> [SELECTED]" if is_chosen else ""
            print(f"    [{c['index']}] {c['name']} {tag}{mark}")

        if chosen_cam:
            selected_source = str(chosen_cam["index"])
            selected_camera_name = chosen_cam["name"]
            if chosen_cam.get("is_external"):
                print(f"[*] Automatically selected external drone camera: Index {selected_source} ({selected_camera_name})")
            else:
                print(f"[*] Selected primary camera: Index {selected_source} ({selected_camera_name})")
        else:
            selected_source = "0"
            selected_camera_name = "Default Camera 0"
    else:
        print(f"[*] Using explicit video source: {selected_source}")

    check_and_clear_port(args.port)

    print(f"[*] Python Interpreter : {python_exe}")
    print(f"[*] Video Input Source : Index {selected_source} ({selected_camera_name})")
    print(f"[*] Command Center URL : http://localhost:{args.port}\n")

    # 1. Start Streamlit Command Center (Primary Mission Hub)
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "registry_app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--theme.base", "light"
    ]
    print("[1/2] Initializing Tactical Command Center (Streamlit)...", flush=True)
    st_proc = subprocess.Popen(streamlit_cmd, cwd=app_dir)

    # Allow Streamlit 2.5 seconds to initialize port
    time.sleep(2.5)

    if not args.no_browser:
        dashboard_url = f"http://localhost:{args.port}"
        print(f"[*] Opening browser: {dashboard_url}", flush=True)
        webbrowser.open(dashboard_url)

    # 2. Function to launch Edge-AI Perception Engine
    detect_proc = None

    def start_detection():
        nonlocal detect_proc
        detect_cmd = [python_exe, "-u", "detect.py", "--source", str(selected_source)]
        if args.no_gui:
            detect_cmd.append("--no-gui")
        print("[2/2] Launching Edge-AI Perception Pipeline (detect.py)...", flush=True)
        print("-----------------------------------------------------------------------------")
        print(" CONTROLS:")
        print("   In Camera Window: [T] Toggle FLIR Thermal | [Q] Close Camera Window")
        print("   In Console:       [Ctrl+C] Shut down entire system")
        print("-----------------------------------------------------------------------------\n", flush=True)
        detect_proc = subprocess.Popen(detect_cmd, cwd=app_dir)

    start_detection()

    def shutdown(sig=None, frame=None):
        print("\n\n[*] Shutting down Autonomous Disaster Response Drone System...", flush=True)
        for name, proc in [("Perception Engine", detect_proc), ("Command Center", st_proc)]:
            if proc and proc.poll() is None:
                print(f"    - Stopping {name} (PID {proc.pid})...", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[OK] System successfully stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor processes independently: Streamlit stays alive even if camera is closed!
    camera_notified = False
    while True:
        try:
            time.sleep(1)
            # Check if camera stopped
            if detect_proc and detect_proc.poll() is not None:
                if not camera_notified:
                    code = detect_proc.poll()
                    print("\n" + "=" * 70, flush=True)
                    if code == 0:
                        print("[!] Camera perception window closed (User pressed 'Q' or closed window).", flush=True)
                    else:
                        print(f"[!] Perception Engine exited with code {code}.", flush=True)
                        print("    - If camera failed to open, check connection or try: python run_system.py --source 0", flush=True)
                    print(f"[*] Tactical Command Center remains ACTIVE at http://localhost:{args.port}", flush=True)
                    print("    - All mission logs, survivor lists, and SITREPs are safely persisted.", flush=True)
                    print("    - Press [Ctrl+C] in this console when you are done to exit.", flush=True)
                    print("=" * 70 + "\n", flush=True)
                    camera_notified = True

            # If Streamlit itself crashed or was killed, then shut down everything
            if st_proc.poll() is not None:
                print("[!] Command center process terminated.", flush=True)
                shutdown()
                break

        except KeyboardInterrupt:
            shutdown()
            break

if __name__ == "__main__":
    main()
