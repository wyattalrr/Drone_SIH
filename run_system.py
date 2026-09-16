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

def main():
    parser = argparse.ArgumentParser(description="Launch the Autonomous Disaster Response Drone System")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to test video)")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit dashboard port (default: 8501)")
    parser.add_argument("--no-gui", action="store_true", help="Run detection headless without OpenCV window")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    print_banner()

    python_exe = get_python_executable()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[*] Python Interpreter : {python_exe}")
    print(f"[*] Video Input Source : {args.source}")
    print(f"[*] Command Center URL : http://localhost:{args.port}\n")

    # 1. Start Streamlit Command Center (Primary Mission Hub)
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "registry_app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--theme.base", "light"
    ]
    print("[1/2] Initializing Tactical Command Center (Streamlit)...")
    st_proc = subprocess.Popen(streamlit_cmd, cwd=app_dir)

    # Allow Streamlit 2.5 seconds to initialize port
    time.sleep(2.5)

    if not args.no_browser:
        dashboard_url = f"http://localhost:{args.port}"
        print(f"[*] Opening browser: {dashboard_url}")
        webbrowser.open(dashboard_url)

    # 2. Function to launch Edge-AI Perception Engine
    detect_proc = None

    def start_detection():
        nonlocal detect_proc
        detect_cmd = [python_exe, "detect.py", "--source", str(args.source)]
        if args.no_gui:
            detect_cmd.append("--no-gui")
        print("[2/2] Launching Edge-AI Perception Pipeline (detect.py)...")
        print("-----------------------------------------------------------------------------")
        print(" CONTROLS:")
        print("   In Camera Window: [T] Toggle FLIR Thermal | [Q] Close Camera Window")
        print("   In Console:       [Ctrl+C] Shut down entire system")
        print("-----------------------------------------------------------------------------\n")
        detect_proc = subprocess.Popen(detect_cmd, cwd=app_dir)

    start_detection()

    def shutdown(sig=None, frame=None):
        print("\n\n[*] Shutting down Autonomous Disaster Response Drone System...")
        for name, proc in [("Perception Engine", detect_proc), ("Command Center", st_proc)]:
            if proc and proc.poll() is None:
                print(f"    - Stopping {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[OK] System successfully stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor processes independently: Streamlit stays alive even if camera is closed!
    camera_notified = False
    while True:
        try:
            time.sleep(1)
            # Check if camera stopped (e.g. user pressed 'Q')
            if detect_proc and detect_proc.poll() is not None:
                if not camera_notified:
                    print("\n" + "="*70)
                    print("[!] Camera perception window closed (User pressed 'Q' or stream ended).")
                    print(f"[*] Tactical Command Center remains ACTIVE at http://localhost:{args.port}")
                    print("    - All mission logs, survivor lists, and SITREPs are safely persisted.")
                    print("    - Press [Ctrl+C] in this console when you are done to exit.")
                    print("="*70 + "\n")
                    camera_notified = True

            # If Streamlit itself crashed or was killed, then shut down everything
            if st_proc.poll() is not None:
                print("[!] Command center process terminated.")
                shutdown()
                break

        except KeyboardInterrupt:
            shutdown()
            break

if __name__ == "__main__":
    main()
