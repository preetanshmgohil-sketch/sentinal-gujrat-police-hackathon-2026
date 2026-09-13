import os
import sys
import re
import io
import csv
import json
import cv2
import time
import math
import random
import base64
import logging
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import sqlite3
import requests
import threading
import numpy as np
from collections import deque
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, Response, jsonify, render_template, request
from cvzone import cornerRect, putTextRect
from ultralytics import YOLO

print("[INFO] RTSP/TCP capture via OpenCV + FFmpeg (forced TCP transport)")

def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

cv2.setNumThreads(1)

app = Flask(__name__)
DB_PATH = 'registry.db'

DEV_MODE = '--dev' in sys.argv
GATEWAY_HOST = os.environ.get("SENTINEL_GATEWAY", "https://live.sentinelgujarat.in")

AUTH_COOKIE_FILE = os.path.join(os.path.dirname(__file__), '.auth_cookie')
_gateway_auth_cookie = ''

def _load_auth_cookie():
    global _gateway_auth_cookie
    try:
        if os.path.exists(AUTH_COOKIE_FILE):
            with open(AUTH_COOKIE_FILE, 'r') as f:
                _gateway_auth_cookie = f.read().strip()
            if _gateway_auth_cookie:
                print(f"[AUTH] Loaded saved gateway cookie ({len(_gateway_auth_cookie)} chars)")
    except Exception:
        pass

def _save_auth_cookie(cookie):
    global _gateway_auth_cookie
    _gateway_auth_cookie = cookie
    with open(AUTH_COOKIE_FILE, 'w') as f:
        f.write(cookie)
    print(f"[AUTH] Saved gateway cookie ({len(cookie)} chars)")

_load_auth_cookie()

def _update_ffmpeg_options():
    opts = "rtsp_transport;tcp|User-Agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    if _gateway_auth_cookie:
        cookie_str = _gateway_auth_cookie if '=' in _gateway_auth_cookie else f'sentinel={_gateway_auth_cookie}'
        opts += f"|headers;Referer: https://cctv.corp8.cloud/\\r\\nCookie: {cookie_str}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts

_update_ffmpeg_options()

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

class GatewayConnectionError(Exception):
    pass

ANALYTICS_ACTIVE = {}
analytics_lock = threading.Lock()
detection_log = deque(maxlen=2000)
analytics_debug_log = deque(maxlen=50)

ENGINE_TIMEOUT = 60.0
engines = {}
engines_lock = threading.Lock()
engines_last_access = {}

def load_yolo_model():
    try:
        import torch
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            print(f"[INFO] YOLOv8 Nano loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] YOLOv8 Nano loaded on CPU")
        return model
    except Exception as e:
        print(f"[ERROR] YOLO model load failed: {e}")
        return None

yolo_model = load_yolo_model()

def get_yolo_device():
    try:
        import torch
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except:
        return 'cpu'

YOLO_DEVICE = get_yolo_device()

def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA cache_size = -2000;")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cctv_registry (
            cam_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT DEFAULT 'Police',
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            codec TEXT DEFAULT 'H264',
            rtsp_url TEXT NOT NULL,
            hls_url TEXT,
            whep_url TEXT,
            fallback_url TEXT,
            status TEXT CHECK(status IN ('ONLINE', 'OFFLINE', 'DEGRADED')) DEFAULT 'ONLINE',
            fps_declared REAL,
            width INTEGER,
            height INTEGER,
            last_pts_ms REAL DEFAULT 0.0,
            grid_slot INTEGER DEFAULT -1,
            analytics_enabled INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cam_status ON cctv_registry (cam_id, status);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_grid_slot ON cctv_registry (grid_slot) WHERE grid_slot >= 0;')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL UNIQUE,
            owner_name TEXT,
            vehicle_type TEXT DEFAULT 'Unknown',
            reason TEXT DEFAULT 'Under Surveillance',
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            cam_id TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            frame_data BLOB,
            is_on_watchlist INTEGER DEFAULT 0,
            alert_triggered INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movement_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            cam_id TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            direction TEXT DEFAULT 'Unknown',
            speed_kmh REAL DEFAULT 0.0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            label TEXT,
            confidence REAL DEFAULT 0.0,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            cam_id TEXT NOT NULL,
            owner_name TEXT,
            reason TEXT,
            priority TEXT,
            confidence REAL DEFAULT 0.0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acknowledged INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violation_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id INTEGER,
            cam_id TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            fine TEXT,
            section TEXT,
            description TEXT,
            confidence REAL DEFAULT 0.0,
            screenshot_path TEXT,
            plate_number TEXT,
            vehicle_type TEXT,
            vehicle_color TEXT,
            vehicle_make TEXT,
            vehicle_model TEXT,
            vehicle_direction TEXT,
            vehicle_action TEXT,
            lighting_condition TEXT,
            weather TEXT,
            road_condition TEXT,
            num_vehicles INTEGER DEFAULT 0,
            num_persons INTEGER DEFAULT 0,
            ai_scene_description TEXT,
            ai_violation_evidence TEXT,
            yolo_detections TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (violation_id) REFERENCES events(id)
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON vehicle_detections (plate_number);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_movement_plate ON movement_history (plate_number);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_plate ON watchlist (plate_number);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_cam ON events (cam_id, timestamp);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type, timestamp);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_ts ON watchlist_alerts (timestamp);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_ack ON watchlist_alerts (acknowledged, timestamp);')
    
    conn.commit()
    conn.close()

def sync_sentinel_catalogue(gateway_host=None):
    if gateway_host is None:
        gateway_host = GATEWAY_HOST

    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            cookies_dict = {}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://cctv.corp8.cloud/',
                'Accept': 'application/json',
            }
            if _gateway_auth_cookie:
                cookie_name = 'sentinel'
                cookie_val = _gateway_auth_cookie
                if '=' in _gateway_auth_cookie:
                    parts = _gateway_auth_cookie.split('=', 1)
                    cookie_name = parts[0]
                    cookie_val = parts[1]
                cookies_dict = {cookie_name: cookie_val}

            cameras_json_url = "https://cctv.corp8.cloud/cameras.json"
            res = requests.get(cameras_json_url, timeout=10.0, verify=False,
                               headers=headers, cookies=cookies_dict)
            if res.status_code != 200:
                print(f"[WARN] Gateway cameras.json returned {res.status_code}, using existing DB")
                return False

            try:
                data = res.json()
            except Exception:
                if '<html' in res.text[:200].lower() or '<!doctype' in res.text[:200].lower():
                    print(f"[WARN] Gateway returned HTML (auth required?), using existing DB cameras")
                    return False
                raise

            cameras = data if isinstance(data, list) else data.get('cameras', [])
            if not cameras or not isinstance(cameras, list):
                print(f"[WARN] Gateway returned no cameras, using existing DB")
                return False

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            saved_grid = {r[0]: (r[1], r[2]) for r in cursor.execute(
                "SELECT cam_id, grid_slot, analytics_enabled FROM cctv_registry WHERE grid_slot >= 0").fetchall()}
            cursor.execute("DELETE FROM cctv_registry")

            coords_map = {
                'junagadh': (21.5222, 70.4591), 'rajkot': (22.3039, 70.8022),
                'gandhidham': (23.2166, 70.2359), 'navsari': (20.9517, 72.9324),
                'bilimora': (20.7689, 72.9594), 'adalaj': (23.1645, 72.5820),
                'visat': (23.2500, 72.6300), 'paldi': (23.0200, 72.5200),
                'ongc': (23.0500, 72.5500), 'char chowk': (21.5300, 70.4700),
                'timbavadi': (21.5500, 70.4000), 'hero showroom': (21.5100, 70.3800),
                'majewadi': (21.5400, 70.4200), 'new bypass': (21.5200, 70.4600),
                'dolatpara': (21.5100, 70.4400), 'mandir': (23.1800, 72.5900),
                'vidhyalaya': (23.0300, 72.5100), 'delight': (23.0100, 72.5000),
                'suvidha': (23.0000, 72.4900), 'dehgam': (23.1000, 72.7000),
                'dhanori': (23.1200, 72.7200), 'tankal': (20.7800, 72.9700),
                'mohanpura': (20.9300, 72.9200), 'patan': (23.8500, 72.1200),
                'mervada': (21.4800, 70.3500), 'kheram': (21.5000, 70.3800),
                'bus port': (22.2900, 70.8100), 'khaparia': (20.9200, 72.9100),
                'circle': (23.1500, 72.6200),
            }
            default_coords = [
                (23.2156, 72.6369), (23.2200, 72.6400), (23.2170, 72.6380),
                (23.2140, 72.6350), (23.2190, 72.6410), (21.5222, 70.4591),
                (21.5100, 70.3800), (21.5400, 70.4200), (21.5200, 70.4600),
                (21.5300, 70.4700), (21.5100, 70.4400), (23.1800, 72.5900),
                (23.0300, 72.5100), (23.0100, 72.5000), (23.0000, 72.4900),
                (23.2500, 72.6300), (22.3039, 70.8022), (22.2900, 70.8100),
                (20.9517, 72.9324), (20.9300, 72.9200), (23.8500, 72.1200),
                (21.4800, 70.3500), (21.5000, 70.3800), (23.1000, 72.7000),
                (23.1200, 72.7200), (20.7800, 72.9700), (20.7689, 72.9594),
                (20.7800, 72.9600), (20.7900, 72.9700), (23.2166, 70.2359),
            ]

            def get_coords(location_str, idx):
                loc = location_str.lower()
                for keyword, coords in coords_map.items():
                    if keyword in loc:
                        return coords
                return default_coords[idx % len(default_coords)]

            for idx, cam in enumerate(cameras):
                cam_id = cam.get('id')
                if not cam_id:
                    continue

                rtsp_url = f"rtsp://preetanshmgohil%40gmail.com:VB82-LHC2-U8VC@103.250.160.189:8554/stream/{cam_id}"
                hls_url = ''
                fallback_url = ''

                location_name = cam.get('name', f'Sentinel Cam {cam_id}')
                lat, lng = get_coords(location_name, idx)

                dept_keywords = {
                    'junagadh': 'Junagadh Police', 'rajkot': 'Rajkot Police',
                    'gandhidham': 'Kutch Police', 'navsari': 'Navsari Police',
                    'bilimora': 'Navsari Police', 'adalaj': 'Gandhinagar Police',
                    'visat': 'Ahmedabad Police', 'paldi': 'Ahmedabad Police',
                    'ongc': 'Ahmedabad Police', 'mandir': 'Gandhinagar Police',
                    'bus port': 'Rajkot Police', 'patan': 'Patan Police',
                    'khaparia': 'Navsari Police', 'mohanpura': 'Navsari Police',
                    'dehgam': 'Kheda Police', 'dhanori': 'Kheda Police',
                    'tankal': 'Navsari Police', 'mervada': 'Junagadh Police',
                    'kheram': 'Junagadh Police',
                }
                department = 'Gujarat Police'
                for kw, dept in dept_keywords.items():
                    if kw in location_name.lower():
                        department = dept
                        break

                width = cam.get('width') or 0
                height = cam.get('height') or 0
                fps = cam.get('fps') or 0.0

                cursor.execute('''
                    INSERT INTO cctv_registry (cam_id, name, department, latitude, longitude, codec, rtsp_url, hls_url, whep_url, fallback_url, status, fps_declared, width, height, grid_slot, analytics_enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, -1, 0)
                ''', (
                    cam_id, location_name, department, lat, lng,
                    cam.get('codec', '') or 'H264',
                    rtsp_url, hls_url, '', fallback_url,
                    'ONLINE',
                    fps if fps else None, width if width else None, height if height else None
                ))

            conn.commit()
            for cid, (slot, ai) in saved_grid.items():
                cursor.execute("UPDATE cctv_registry SET grid_slot=?, analytics_enabled=? WHERE cam_id=?", (slot, ai, cid))
            conn.commit()
            conn.close()
            print(f"[INFO] Registry synced with Sentinel Catalogue: {len(cameras)} cameras loaded.")
            return True

        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Gateway connection failed: {e}")
        except requests.exceptions.Timeout as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Gateway request timed out: {e}")
        except GatewayConnectionError as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: {e}")
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Unexpected error: {e}")

        if attempt < max_retries - 1:
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1.0)
            print(f"[INFO] Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

    if DEV_MODE:
        print("[DEV MODE] Gateway unreachable. Loading local test cameras.")
        _load_dev_cameras()
        return True

    print("[WARN] Gateway unreachable. Loading dev cameras as fallback.")
    _load_dev_cameras()
    return True

def _load_dev_cameras():
    dev_cameras = [
        ('CAM-01', 'Sector 18 Entry Gate', 'Police Dept', 23.2156, 72.6369, 'H264', 'assets/traffic.mp4', 'ONLINE'),
        ('CAM-02', 'Bus Terminal North', 'Transport Dept', 23.2170, 72.6380, 'H264', 'assets/pedestrians.mp4', 'ONLINE'),
        ('CAM-03', 'Civic Center Junction', 'Municipal Corp', 23.2140, 72.6350, 'H265', 'assets/traffic.mp4', 'ONLINE'),
        ('CAM-04', 'Highway Patrol Post 4', 'Traffic Police', 23.2190, 72.6400, 'H264', 'assets/pedestrians.mp4', 'ONLINE'),
    ]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cctv_registry")
    for cam in dev_cameras:
        cursor.execute('''
            INSERT INTO cctv_registry (cam_id, name, department, latitude, longitude, codec, rtsp_url, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', cam)
    conn.commit()
    conn.close()
    print(f"[DEV MODE] Loaded {len(dev_cameras)} local test cameras.")

class SentinelStreamEngine:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_bytes = None
        self.current_pts = 0.0
        self.is_discontinuity = False
        self.lock = threading.Lock()
        self.running = True
        self.last_access = time.time()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        row = cursor.execute("SELECT rtsp_url, hls_url, fallback_url, width, height FROM cctv_registry WHERE cam_id=?", (cam_id,)).fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Camera {cam_id} missing from Registry.")
        
        self.stream_url = row[0]
        self.hls_url = row[1] or ''
        self.fallback_url = row[2] or ''
        self.cam_width = row[3] if row[3] else 640
        self.cam_height = row[4] if row[4] else 480
        
        self.out_width = min(self.cam_width, 640)
        self.out_height = min(self.cam_height, 480)
        if self.out_width % 2 != 0:
            self.out_width -= 1
        if self.out_height % 2 != 0:
            self.out_height -= 1
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _capture_loop(self):
        backoff_attempts = 0
        cap = None
        consecutive_read_fails = 0
        MAX_READ_FAILS_BEFORE_RECONNECT = 200
        # Try RTSP (TCP) first per reference spec, fall back to HLS
        urls_to_try = [u for u in [self.stream_url, self.hls_url] if u]
        url_idx = 0

        while self.running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                    cap = None

                backoff_time = min(30.0, 1.5 * math.pow(1.5, backoff_attempts)) + random.uniform(0, 0.5)
                time.sleep(backoff_time)
                backoff_attempts += 1

                if not urls_to_try:
                    time.sleep(10)
                    continue

                try_url = urls_to_try[url_idx % len(urls_to_try)]
                is_rtsp = try_url.startswith('rtsp://')
                proto_label = 'RTSP/TCP' if is_rtsp else 'HLS'

                print(f"[INFO] {self.cam_id}: Connecting via {proto_label}: {try_url}")
                cap = cv2.VideoCapture(try_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    cap.release()
                    cap = None
                    consecutive_read_fails = 0
                    url_idx += 1  # try next URL on next backoff
                    continue

                print(f"[INFO] {self.cam_id}: Connected via {proto_label}")
                url_idx = 0  # reset on success — prefer RTSP next time

            prev_pts = -1.0
            consecutive_read_fails = 0

            while self.running and cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    consecutive_read_fails += 1
                    if consecutive_read_fails >= MAX_READ_FAILS_BEFORE_RECONNECT:
                        print(f"[INFO] {self.cam_id}: Feed read failed {MAX_READ_FAILS_BEFORE_RECONNECT}x, reconnecting")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_read_fails = 0
                backoff_attempts = 0
                # Drive timing from PTS per reference spec, not wall clock
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pts <= 0:
                    pts = time.time() * 1000.0
                delta_pts = pts - prev_pts
                # Scene discontinuity (loop restart or reboot) per reference
                discontinuity = (delta_pts < 0 or delta_pts > 5000.0) if prev_pts >= 0 else False
                prev_pts = pts

                resized = cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR)
                _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 55])

                with self.lock:
                    self.frame_bytes = buffer.tobytes()
                    self.current_pts = pts
                    self.is_discontinuity = discontinuity
                    self.last_access = time.time()

            if cap is not None:
                cap.release()
                cap = None

    def generate_mjpeg(self):
        try:
            while self.running:
                with self.lock:
                    self.last_access = time.time()
                    if self.frame_bytes is None:
                        time.sleep(0.01)
                        continue
                    frame_data = self.frame_bytes

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                time.sleep(1.0 / 15.0)
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[WARN] {self.cam_id}: MJPEG generator error: {e}")

from anpr import ANPREngine as _ANPREngine, PlateResult, log_vehicle_detection as _anpr_log_detection, check_watchlist_and_alert as _anpr_check_watchlist

class ANPREngine:
    def __init__(self):
        self._engine = _ANPREngine(backend="easyocr", min_confidence=0.25)
        self.reader = True
        self.plate_model = None
        self._plate_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
        if os.path.exists(self._plate_model_path):
            try:
                self.plate_model = YOLO(self._plate_model_path)
                print("[INFO] ANPR: YOLO plate detector + EasyOCR")
            except Exception as e:
                print("[WARN] ANPR: plate model load failed:", e)
        print("[INFO] ANPR: Ready (EasyOCR)" + (" + YOLO" if self.plate_model else ""))

    def detect_plate(self, frame):
        return None, 0.0

    def detect_plate_from_bbox(self, frame, bbox):
        return None

    def detect_plates_fullframe(self, frame):
        return []

    def _is_valid_plate_text(self, text):
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)
        if len(text) < 6 or len(text) > 12:
            return False
        if re.fullmatch(r'\d+', text):
            return False
        if re.fullmatch(r'[A-Z]+', text):
            return False
        indian = re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$', text)
        bh = re.match(r'^\d{2}BH\d{4}[A-Z]{1,2}$', text)
        if indian or bh:
            return True
        if len(text) >= 8 and re.match(r'^[A-Z]{2}\d', text):
            return True
        return False

    def _enhance_plate_crop(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

anpr_engine = ANPREngine()

try:
    from scene_analyzer import SceneAnalyzer as _SceneAnalyzer
    scene_analyzer = _SceneAnalyzer(db_path=DB_PATH)
    print("[INFO] Scene Analyzer: NIM VLM (meta/llama-3.2-11b-vision-instruct) ready (lazy-load on first use)")
except Exception as e:
    scene_analyzer = None
    print(f"[WARN] Scene Analyzer not available: {e}")

import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def compute_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360

def bearing_to_compass(brng):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return dirs[int((brng + 11.25) / 22.5) % 16]

def compute_movement_info(prev_lat, prev_lon, prev_time, curr_lat, curr_lon, curr_time):
    dist = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
    if prev_time and curr_time:
        try:
            t1 = datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S')
            t2 = datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            try:
                t1 = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return 'Unknown', 0.0
        secs = (t2 - t1).total_seconds()
        if secs > 0 and dist > 0.001:
            speed = (dist / secs) * 3600
            brng = compute_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
            return bearing_to_compass(brng), round(speed, 1)
    return 'Unknown', 0.0

def log_vehicle_detection(plate_number, cam_id, confidence, is_on_watchlist=False, alert_triggered=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO vehicle_detections (plate_number, cam_id, confidence, is_on_watchlist, alert_triggered) VALUES (?, ?, ?, ?, ?)",
        (plate_number, cam_id, confidence, 1 if is_on_watchlist else 0, 1 if alert_triggered else 0)
    )
    conn.commit()
    conn.close()

def log_movement(plate_number, cam_id, latitude, longitude, direction='Unknown', speed_kmh=0.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    existing = cursor.execute(
        """SELECT id, timestamp FROM movement_history
           WHERE plate_number=? AND cam_id=?
           ORDER BY timestamp DESC LIMIT 1""",
        (plate_number, cam_id)
    ).fetchone()

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if existing:
        try:
            last_ts = datetime.strptime(existing[1], '%Y-%m-%d %H:%M:%S')
            diff = (datetime.now() - last_ts).total_seconds()
            if diff < 60:
                direction, speed_kmh = compute_movement_info(
                    latitude, longitude, existing[1], latitude, longitude, now_str
                )
                cursor.execute(
                    "UPDATE movement_history SET timestamp=?, direction=?, speed_kmh=? WHERE id=?",
                    (now_str, direction, speed_kmh, existing[0])
                )
                conn.commit()
                conn.close()
                return
        except (ValueError, TypeError):
            pass

    prev = cursor.execute(
        """SELECT latitude, longitude, timestamp FROM movement_history
           WHERE plate_number=? AND timestamp < ?
           ORDER BY timestamp DESC LIMIT 1""",
        (plate_number, now_str)
    ).fetchone()

    if prev and prev[0] and prev[1] and latitude and longitude:
        direction, speed_kmh = compute_movement_info(
            prev[0], prev[1], prev[2], latitude, longitude, now_str
        )

    cursor.execute(
        "INSERT INTO movement_history (plate_number, cam_id, latitude, longitude, direction, speed_kmh) VALUES (?, ?, ?, ?, ?, ?)",
        (plate_number, cam_id, latitude, longitude, direction, speed_kmh)
    )
    conn.commit()
    conn.close()

def get_camera_location(cam_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    result = cursor.execute("SELECT latitude, longitude FROM cctv_registry WHERE cam_id=?", (cam_id,)).fetchone()
    conn.close()
    return result if result else (0.0, 0.0)

def log_event(cam_id, event_type, label, confidence=0.0, details=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (cam_id, event_type, label, confidence, details) VALUES (?, ?, ?, ?, ?)",
        (cam_id, event_type, label, confidence, details)
    )
    conn.commit()
    conn.close()

def check_watchlist(plate_number):
    if not plate_number:
        return None
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cleaned = re.sub(r'[^A-Z0-9]', '', str(plate_number).upper())
    match = cursor.execute(
        "SELECT id, owner_name, vehicle_type, reason, priority FROM watchlist WHERE plate_number=? AND is_active=1",
        (cleaned,)
    ).fetchone()
    if not match and len(cleaned) > 4:
        match = cursor.execute(
            "SELECT id, owner_name, vehicle_type, reason, priority FROM watchlist WHERE plate_number LIKE ? AND is_active=1",
            (f"%{cleaned[2:-2]}%",)
        ).fetchone()
    conn.close()
    return match

def log_watchlist_alert(plate_number, cam_id, owner_name, reason, priority, confidence):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO watchlist_alerts (plate_number, cam_id, owner_name, reason, priority, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (plate_number, cam_id, owner_name, reason, priority, confidence)
    )
    conn.commit()
    conn.close()

class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _assign_ids(self, centroids):
        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        distance_matrix = np.zeros((len(object_centroids), len(centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, nc in enumerate(centroids):
                distance_matrix[i, j] = math.hypot(oc[0] - nc[0], oc[1] - nc[1])

        used_rows = set()
        used_cols = set()
        assignments = []
        if distance_matrix.size > 0:
            flat_idx = np.argsort(distance_matrix, axis=None)
            for idx in flat_idx:
                r = idx // len(centroids)
                c = idx % len(centroids)
                if r in used_rows or c in used_cols:
                    continue
                if distance_matrix[r, c] > self.max_distance:
                    break
                assignments.append((object_ids[r], c))
                used_rows.add(r)
                used_cols.add(c)

        unmatched_objects = [object_ids[i] for i in range(len(object_ids)) if i not in {a[0] for a in assignments}]
        for oid in unmatched_objects:
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]
                del self.disappeared[oid]

        for oid, cidx in assignments:
            self.objects[oid] = centroids[cidx]
            self.disappeared[oid] = 0

        unmatched_centroids = [i for i in range(len(centroids)) if i not in used_cols]
        for cidx in unmatched_centroids:
            self.objects[self.next_id] = centroids[cidx]
            self.disappeared[self.next_id] = 0
            self.next_id += 1

    def update(self, bboxes):
        if len(bboxes) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return {}

        centroids = []
        for (x1, y1, x2, y2) in bboxes:
            centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        self._assign_ids(centroids)

        results = {}
        for (x1, y1, x2, y2) in bboxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best_id = None
            best_dist = float('inf')
            for oid, oc in self.objects.items():
                d = math.hypot(oc[0] - cx, oc[1] - cy)
                if d < best_dist:
                    best_dist = d
                    best_id = oid
            if best_id is not None and best_dist < self.max_distance:
                results[tuple([x1, y1, x2, y2])] = best_id
        return results

class AnalyticsEngine:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.last_detections = []
        self.last_access = time.time()
        self.frame_bytes = None
        self.prev_pts = -1.0
        self.discontinuity_detected = False
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=100)
        self.track_history = {}
        self.last_raw_frame = None
        self.last_fullres_frame = None

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        row = cursor.execute("SELECT rtsp_url, hls_url, fallback_url, width, height FROM cctv_registry WHERE cam_id=?", (cam_id,)).fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Camera {cam_id} not found.")
        
        self.stream_url = row[0]
        self.hls_url = row[1] or ''
        self.fallback_url = row[2] or ''
        self.cam_width = row[3] if row[3] else 640
        self.cam_height = row[4] if row[4] else 480
        
        self.out_width = min(self.cam_width, 640)
        self.out_height = min(self.cam_height, 480)
        if self.out_width % 2 != 0:
            self.out_width -= 1
        if self.out_height % 2 != 0:
            self.out_height -= 1

        self.last_annotated = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _capture_loop(self):
        cap = None
        backoff_attempts = 0
        consecutive_read_fails = 0
        total_frames_on_url = 0
        MAX_READ_FAILS_BEFORE_RECONNECT = 200
        # RTSP (TCP forced) first per reference — AI inference endpoint
        urls_to_try = [u for u in [self.stream_url, self.hls_url] if u]
        url_idx = 0

        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                        cap = None

                    backoff_time = min(30.0, 1.5 * math.pow(1.5, backoff_attempts)) + random.uniform(0, 0.5)
                    time.sleep(backoff_time)
                    backoff_attempts += 1

                    if not urls_to_try:
                        time.sleep(10)
                        continue

                    try_url = urls_to_try[url_idx % len(urls_to_try)]
                    proto_label = 'RTSP/TCP' if try_url.startswith('rtsp://') else 'HLS'

                    analytics_debug_log.append(f"{self.cam_id}: Connecting via {proto_label}: {try_url}")
                    cap = cv2.VideoCapture(try_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        consecutive_read_fails = 0
                        total_frames_on_url = 0
                        url_idx += 1  # try next URL on next attempt
                        continue

                    self.prev_pts = -1.0
                    self.discontinuity_detected = False
                    consecutive_read_fails = 0
                    total_frames_on_url = 0
                    url_idx = 0  # prefer RTSP next time
                    analytics_debug_log.append(f"{self.cam_id}: CONNECTED via {proto_label}")

                frame_skip_counter = 0

                while self.running and cap is not None and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        consecutive_read_fails += 1
                        if consecutive_read_fails >= MAX_READ_FAILS_BEFORE_RECONNECT:
                            analytics_debug_log.append(f"{self.cam_id}: HLS read failed {MAX_READ_FAILS_BEFORE_RECONNECT}x, reconnecting")
                            break
                        time.sleep(0.1)
                        continue

                    consecutive_read_fails = 0
                    backoff_attempts = 0
                    total_frames_on_url += 1
                    pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pts <= 0:
                        pts = time.time() * 1000.0
                    delta_pts = pts - self.prev_pts
                    discontinuity = (delta_pts < 0 or delta_pts > 5000.0) if self.prev_pts >= 0 else False
                    self.prev_pts = pts

                    if discontinuity:
                        self.discontinuity_detected = True

                    frame_skip_counter += 1
                    if frame_skip_counter % 2 == 0:
                        self._process_frame(frame, frame_skip_counter)

                if cap is not None:
                    cap.release()
                    cap = None

            except Exception as e:
                analytics_debug_log.append(f"{self.cam_id}: ERROR {e}")
                if cap is not None:
                    cap.release()
                    cap = None
                total_frames_on_url = 0

    def _process_frame(self, frame, frame_skip_counter):
        with analytics_lock:
            is_active = ANALYTICS_ACTIVE.get(self.cam_id, False)

        run_yolo = is_active and yolo_model is not None
        detections = []
        vehicle_boxes = []
        annotated = None
        scale_x = 1.0
        scale_y = 1.0

        if run_yolo:
            if self.discontinuity_detected:
                self.last_detections = []
                self.tracker = CentroidTracker(max_disappeared=30, max_distance=100)
                self.track_history = {}
                self.discontinuity_detected = False

            small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            scale_x = frame.shape[1] / 640.0
            scale_y = frame.shape[0] / 480.0
            with self.lock:
                self.last_raw_frame = small.copy()
                self.last_fullres_frame = frame.copy() if frame.shape[0] <= 1080 else small.copy()
            avg_brightness = cv2.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))[0]
            if avg_brightness < 80:
                small = enhance_low_light(small)
            results = yolo_model.predict(small, classes=[0, 1, 2, 3, 5, 7], conf=0.20, verbose=False, device=YOLO_DEVICE)
            annotated = small.copy()
            VEHICLE_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            CLASS_COLORS = {
                'person': (0, 200, 255), 'car': (0, 255, 0), 'bicycle': (255, 200, 0),
                'motorcycle': (0, 150, 255), 'bus': (255, 255, 0), 'truck': (255, 100, 0),
            }
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = VEHICLE_CLASSES.get(cls, 'vehicle')
                    color = CLASS_COLORS.get(label, (255, 255, 255))
                    detections.append({'class': label, 'confidence': round(conf, 2)})
                    if cls in VEHICLE_CLASSES and cls != 0:
                        vehicle_boxes.append((x1, y1, x2, y2))

            tracked = self.tracker.update(vehicle_boxes) if vehicle_boxes else {}
            for (x1, y1, x2, y2), track_id in tracked.items():
                label = None
                for det_idx, vbox in enumerate(vehicle_boxes):
                    if vbox == (x1, y1, x2, y2):
                        label = detections[det_idx]['class'] if det_idx < len(detections) else 'vehicle'
                        break
                if label is None:
                    label = 'vehicle'

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = CLASS_COLORS.get(label, (255, 255, 255))
                cornerRect(annotated, (x1, y1, x2, y2), l=20, t=2, colorR=color, colorC=(255, 255, 255))
                putTextRect(annotated, f"#{track_id} {label.upper()}", (x1, y1 - 35),
                            scale=0.5, thickness=1, colorT=(0, 0, 0), colorR=color)

                if track_id not in self.track_history:
                    self.track_history[track_id] = {'positions': [], 'label': label, 'first_seen': time.time()}
                self.track_history[track_id]['positions'].append((cx, cy, time.time()))
                if len(self.track_history[track_id]['positions']) > 50:
                    self.track_history[track_id]['positions'] = self.track_history[track_id]['positions'][-50:]

            if detections and frame_skip_counter % 10 == 0:
                for det in detections[:10]:
                    log_event(self.cam_id, 'yolo', det['class'], det['confidence'])

        with self.lock:
            self.last_detections = detections
            self.frame_count += 1
            self.last_access = time.time()

        if detections and frame_skip_counter % 10 == 0:
            detection_log.append({
                'cam_id': self.cam_id,
                'frame': self.frame_count,
                'detections': detections,
                'timestamp': time.time()
            })

        if run_yolo and annotated is not None:
            self.last_annotated = annotated

        if self.last_annotated is not None:
            resized = cv2.resize(self.last_annotated, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR)
            ret, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                with self.lock:
                    self.frame_bytes = buf.tobytes()

    def generate_analytics_feed(self):
        try:
            while self.running:
                with self.lock:
                    self.last_access = time.time()
                    if self.frame_bytes is None:
                        time.sleep(0.01)
                        continue
                    frame_data = self.frame_bytes

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                time.sleep(1.0 / 15.0)
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[WARN] {self.cam_id}: Analytics feed generator error: {e}")

init_db()
sync_sentinel_catalogue()

analytics_engines = {}

def engine_gc_loop():
    while True:
        time.sleep(15.0)
        now = time.time()

        with engines_lock:
            stalecams = []
            for cid, t in engines_last_access.items():
                age = now - t
                if age <= ENGINE_TIMEOUT:
                    continue
                engine = engines.get(cid)
                if engine and engine.running:
                    engines_last_access[cid] = now
                    continue
                stalecams.append(cid)

            for cam_id in stalecams:
                if cam_id in engines:
                    engines[cam_id].stop()
                    del engines[cam_id]
                if cam_id in engines_last_access:
                    del engines_last_access[cam_id]
                print(f"[GC] Released idle streaming engine for {cam_id}")

        conn = sqlite3.connect(DB_PATH)
        grid_cameras = set(row[0] for row in conn.execute(
            "SELECT cam_id FROM cctv_registry WHERE grid_slot >= 0"
        ).fetchall())
        conn.close()

        with engines_lock:
            for cam_id in list(analytics_engines.keys()):
                with analytics_lock:
                    is_active = ANALYTICS_ACTIVE.get(cam_id, False)
                if not is_active or cam_id not in grid_cameras:
                    try:
                        analytics_engines[cam_id].stop()
                    except Exception:
                        pass
                    del analytics_engines[cam_id]
                    with analytics_lock:
                        ANALYTICS_ACTIVE[cam_id] = False
                    print(f"[GC] Stopped analytics for camera {cam_id} (active={is_active}, in_grid={cam_id in grid_cameras})")

gc_thread = threading.Thread(target=engine_gc_loop, daemon=True)
gc_thread.start()

conn = sqlite3.connect(DB_PATH)
startup_grid = conn.execute("SELECT cam_id, analytics_enabled FROM cctv_registry WHERE grid_slot >= 0").fetchall()
conn.close()
for cam_id, ai_enabled in startup_grid:
    with engines_lock:
        if cam_id not in engines:
            try:
                engine = SentinelStreamEngine(cam_id)
                engines[cam_id] = engine
                engines_last_access[cam_id] = time.time()
                print(f"[STARTUP] Restored streaming engine for camera {cam_id}")
            except Exception as e:
                print(f"[STARTUP] Failed to restore streaming for {cam_id}: {e}")
    if ai_enabled:
        with engines_lock:
            if cam_id not in analytics_engines:
                try:
                    aengine = AnalyticsEngine(cam_id)
                    analytics_engines[cam_id] = aengine
                    with analytics_lock:
                        ANALYTICS_ACTIVE[cam_id] = True
                    print(f"[STARTUP] Restored analytics engine for camera {cam_id}")
                except Exception as e:
                    print(f"[STARTUP] Failed to restore analytics for {cam_id}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/watchlist')
def watchlist_page():
    return render_template('watchlist.html')

@app.route('/vehicle-search')
def vehicle_search_page():
    return render_template('vehicle_search.html')

@app.route('/anpr-alerts')
def anpr_alerts_page():
    return render_template('anpr_alerts.html')

@app.route('/events')
def events_page():
    return render_template('events.html')

@app.route('/violations')
def violations_page():
    return render_template('violations.html')

@app.route('/api/scene/analyze', methods=['POST'])
def scene_analyze():
    if scene_analyzer is None:
        return jsonify({"error": "Scene analyzer not available"}), 503
    data = request.get_json(silent=True)
    cam_id = data.get("cam_id") if data else None
    if not cam_id:
        return jsonify({"error": "cam_id required"}), 400
    frame = None
    with engines_lock:
        engine = engines.get(cam_id)
    if engine is not None:
        with engine.lock:
            fb = engine.frame_bytes
        if fb is not None:
            buf = np.frombuffer(fb, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        with engines_lock:
            aengine = analytics_engines.get(cam_id)
        if aengine is not None:
            with aengine.lock:
                fb = aengine.frame_bytes
            if fb is not None:
                buf = np.frombuffer(fb, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute("SELECT fallback_url, hls_url FROM cctv_registry WHERE cam_id=?", (cam_id,)).fetchone()
            conn.close()
            if row:
                url = row[0] or row[1]
                if url:
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        ok, frame = cap.read()
                        cap.release()
        except Exception:
            pass
    if frame is None:
        return jsonify({"error": "No frame available"}), 404
    desc = scene_analyzer.describe_scene(frame, cam_id)
    return jsonify({
        "description": desc.description,
        "lighting": desc.lighting,
        "weather": desc.weather,
        "road_condition": desc.road_condition,
        "processing_time_ms": round(desc.processing_time_ms, 1),
        "camera_id": cam_id,
    })

@app.route('/api/scene/violations', methods=['POST'])
def scene_violations():
    if scene_analyzer is None:
        return jsonify({"error": "Scene analyzer not available"}), 503
    data = request.get_json(silent=True)
    cam_id = data.get("cam_id") if data else None
    if not cam_id:
        return jsonify({"error": "cam_id required"}), 400
    with engines_lock:
        engine = engines.get(cam_id)
    if engine is None:
        return jsonify({"error": f"No stream engine for {cam_id}"}), 404
    with engine.lock:
        fb = engine.frame_bytes
    if fb is None:
        return jsonify({"error": "No frame available"}), 404
    buf = np.frombuffer(fb, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Failed to decode frame"}), 404
    yolo_dets = []
    plate_num = None
    with engines_lock:
        aengine = analytics_engines.get(cam_id)
    if aengine:
        with aengine.lock:
            yolo_dets = aengine.last_detections or []
            for d in yolo_dets:
                if d.get("class") == "plate" and d.get("plate_number"):
                    plate_num = d["plate_number"]
                    break
    violations = scene_analyzer.detect_violations(frame, cam_id, yolo_dets, plate_num)
    return jsonify({
        "violations": [
            {"rule_id": v.rule_id, "rule_name": v.rule_name, "severity": v.severity,
             "fine": v.fine, "section": v.section, "description": v.description,
             "confidence": v.confidence, "vehicle_type": v.vehicle_type,
             "vehicle_color": v.vehicle_color, "vehicle_make": v.vehicle_make,
             "vehicle_direction": v.vehicle_direction, "vehicle_action": v.vehicle_action,
             "plate_number": v.plate_number, "lighting": v.lighting_condition,
             "weather": v.weather, "road_condition": v.road_condition,
             "screenshot_path": v.screenshot_path,
             "ai_evidence": v.ai_violation_evidence,
             "scene_description": v.ai_scene_description}
            for v in violations
        ],
        "camera_id": cam_id,
        "count": len(violations),
    })

@app.route('/api/scene/violations/recent')
def scene_violations_recent():
    if scene_analyzer is None:
        return jsonify([])
    limit = request.args.get("limit", 50, type=int)
    return jsonify(scene_analyzer.get_recent_violations(limit))

@app.route('/api/violation/screenshot/<int:violation_id>')
def violation_screenshot(violation_id):
    if scene_analyzer is None:
        return "Scene analyzer not available", 503
    path = scene_analyzer.get_violation_screenshot(violation_id)
    if path is None:
        return "Screenshot not found", 404
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return Response(f.read(), mimetype=mime)

@app.route('/api/scene/status')
def scene_status():
    return jsonify({
        "model_loaded": scene_analyzer.is_loaded() if scene_analyzer else False,
        "available": scene_analyzer is not None,
        "model_id": "meta/llama-3.2-11b-vision-instruct",
    })

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cams = [dict(row) for row in cursor.execute("SELECT * FROM cctv_registry").fetchall()]
    conn.close()
    return jsonify(cams)

@app.route('/api/cameras/<cam_id>', methods=['GET'])
def get_camera(cam_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cam = cursor.execute("SELECT * FROM cctv_registry WHERE cam_id=?", (cam_id,)).fetchone()
    conn.close()
    if cam:
        return jsonify(dict(cam))
    return jsonify({"error": "Camera not found"}), 404

@app.route('/api/cameras/<cam_id>/status', methods=['PUT'])
def update_camera_status(cam_id):
    data = request.get_json(silent=True)
    if not data or 'status' not in data:
        return jsonify({"error": "Invalid or missing JSON payload, expected 'status' key"}), 400
    new_status = data.get('status')
    if new_status not in ('ONLINE', 'OFFLINE', 'DEGRADED'):
        return jsonify({"error": "Invalid status. Must be ONLINE, OFFLINE, or DEGRADED"}), 400
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE cctv_registry SET status=?, updated_at=CURRENT_TIMESTAMP WHERE cam_id=?", (new_status, cam_id))
    conn.commit()
    conn.close()
    return jsonify({"message": f"Camera {cam_id} status updated to {new_status}"})

@app.route('/api/grid/assign', methods=['POST'])
def assign_grid_slot():
    data = request.get_json(silent=True)
    if not data or 'cam_id' not in data or 'slot' not in data:
        return jsonify({"error": "Invalid or missing JSON payload, expected 'cam_id' and 'slot' keys"}), 400
    
    cam_id = data.get('cam_id')
    slot = data.get('slot')
    
    if not isinstance(slot, int) or slot < 0 or slot > 3:
        return jsonify({"error": "Invalid slot. Must be integer 0-3"}), 400
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE cctv_registry SET grid_slot=-1 WHERE grid_slot=?", (slot,))
    cursor.execute("UPDATE cctv_registry SET grid_slot=?, updated_at=CURRENT_TIMESTAMP WHERE cam_id=?", (slot, cam_id))
    conn.commit()
    conn.close()
    return jsonify({"message": f"Camera {cam_id} assigned to slot {slot}"})

@app.route('/api/grid/cameras', methods=['GET'])
def get_grid_cameras():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT cam_id, name, department, grid_slot, analytics_enabled, status FROM cctv_registry WHERE grid_slot >= 0").fetchall()
    conn.close()
    cameras = []
    for r in rows:
        cameras.append({
            "cam_id": r["cam_id"],
            "name": r["name"],
            "department": r["department"],
            "grid_slot": r["grid_slot"],
            "analytics_enabled": bool(r["analytics_enabled"]),
            "status": r["status"],
        })
    return jsonify({"cameras": cameras})

@app.route('/api/grid/clear/<int:slot>', methods=['POST'])
def clear_grid_slot(slot):
    if slot < 0 or slot > 3:
        return jsonify({"error": "Invalid slot. Must be 0-3"}), 400
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE cctv_registry SET grid_slot=-1, updated_at=CURRENT_TIMESTAMP WHERE grid_slot=?", (slot,))
    conn.commit()
    conn.close()
    return jsonify({"message": f"Slot {slot} cleared"})

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    with engines_lock:
        if cam_id not in engines:
            try:
                engines[cam_id] = SentinelStreamEngine(cam_id)
            except Exception as e:
                return jsonify({"error": str(e)}), 404
        engines_last_access[cam_id] = time.time()
    
    return Response(
        engines[cam_id].generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/analytics_feed/<cam_id>')
def analytics_feed(cam_id):
    with engines_lock:
        if cam_id not in analytics_engines:
            try:
                analytics_engines[cam_id] = AnalyticsEngine(cam_id)
            except Exception as e:
                return jsonify({"error": str(e)}), 404
        engines_last_access[cam_id] = time.time()
    
    return Response(
        analytics_engines[cam_id].generate_analytics_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/auth/cookie', methods=['POST'])
def set_auth_cookie():
    data = request.get_json(silent=True)
    if not data or 'cookie' not in data:
        return jsonify({"error": "Provide {'cookie': '<session_cookie_value>'}"}), 400
    cookie_val = data['cookie'].strip()
    _save_auth_cookie(cookie_val)
    _update_ffmpeg_options()
    return jsonify({"status": "ok", "message": f"Cookie saved ({len(cookie_val)} chars). Restart server to apply to existing engines."})

@app.route('/api/auth/cookie', methods=['GET'])
def get_auth_cookie_status():
    has_cookie = bool(_gateway_auth_cookie)
    return jsonify({"has_cookie": has_cookie, "cookie_length": len(_gateway_auth_cookie) if has_cookie else 0})

@app.route('/api/toggle_analytics', methods=['POST'])
def toggle_analytics():
    data = request.get_json(silent=True)
    if not data or 'cam_id' not in data:
        return jsonify({"error": "Invalid or missing JSON payload, expected 'cam_id' key"}), 400

    cam_id = data.get('cam_id')

    with analytics_lock:
        current = ANALYTICS_ACTIVE.get(cam_id, False)
        ANALYTICS_ACTIVE[cam_id] = not current
        new_state = ANALYTICS_ACTIVE[cam_id]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE cctv_registry SET analytics_enabled=? WHERE cam_id=?", (1 if new_state else 0, cam_id))
    conn.commit()
    conn.close()

    if new_state:
        with engines_lock:
            if cam_id not in analytics_engines:
                try:
                    analytics_engines[cam_id] = AnalyticsEngine(cam_id)
                except Exception as e:
                    with analytics_lock:
                        ANALYTICS_ACTIVE[cam_id] = False
                    return jsonify({"error": str(e)}), 500
    else:
        with engines_lock:
            if cam_id in analytics_engines:
                analytics_engines[cam_id].stop()
                del analytics_engines[cam_id]

    return jsonify({
        "status": "success",
        "cam_id": cam_id,
        "analytics_enabled": new_state,
        "message": f"Analytics for {cam_id} turned {'ON' if new_state else 'OFF'}"
    })

@app.route('/api/analytics/status')
def analytics_status():
    with analytics_lock:
        active_cams = [cid for cid, active in ANALYTICS_ACTIVE.items() if active]
    with engines_lock:
        engine_keys = list(analytics_engines.keys())
        engine_info = {}
        for cid, eng in analytics_engines.items():
            with eng.lock:
                engine_info[cid] = {
                    'frame_count': eng.frame_count,
                    'has_frame_bytes': eng.frame_bytes is not None,
                    'has_annotated': eng.last_annotated is not None,
                    'running': eng.running,
                    'last_detections_count': len(eng.last_detections),
                }
    return jsonify({
        "active_cameras": active_cams,
        "analytics_engines": engine_keys,
        "engine_info": engine_info,
        "debug_log": list(analytics_debug_log),
        "yolo_loaded": yolo_model is not None,
        "device": YOLO_DEVICE,
        "detection_count": len(detection_log)
    })

@app.route('/api/analytics/detections')
def get_detections():
    cam_id = request.args.get('cam_id')
    if cam_id:
        filtered = [d for d in detection_log if d['cam_id'] == cam_id]
        return jsonify(filtered[-20:])
    return jsonify(list(detection_log)[-20:])

@app.route('/api/events')
def get_events():
    limit = request.args.get('limit', 100, type=int)
    event_type = request.args.get('type')
    cam_id = request.args.get('cam_id')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT e.*, c.name as camera_name, c.department FROM events e LEFT JOIN cctv_registry c ON e.cam_id = c.cam_id"
    conditions = []
    params = []

    if event_type:
        types = request.args.getlist('type')
        if len(types) > 1:
            placeholders = ','.join('?' * len(types))
            conditions.append(f"e.event_type IN ({placeholders})")
            params.extend(types)
        else:
            conditions.append("e.event_type = ?")
            params.append(event_type)
    if cam_id:
        conditions.append("e.cam_id = ?")
        params.append(cam_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY e.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/watchlist/alerts')
def get_watchlist_alerts():
    limit = request.args.get('limit', 50, type=int)
    unack_only = request.args.get('unack_only', 'false').lower() == 'true'

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT a.*, c.name as camera_name, c.department FROM watchlist_alerts a LEFT JOIN cctv_registry c ON a.cam_id = c.cam_id"
    params = []

    if unack_only:
        query += " WHERE a.acknowledged = 0"

    query += " ORDER BY a.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/watchlist/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE watchlist_alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    items = [dict(row) for row in cursor.execute("SELECT * FROM watchlist WHERE is_active=1").fetchall()]
    conn.close()
    
    priority_names = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
    for item in items:
        item['priority'] = priority_names.get(item.get('priority', 1), 'low')
    
    return jsonify(items)

@app.route('/api/watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.get_json(silent=True)
    if not data or 'plate_number' not in data:
        return jsonify({"error": "Invalid or missing JSON payload, expected 'plate_number' key"}), 400
    
    plate_number = data.get('plate_number').upper()
    owner_name = data.get('owner_name', 'Unknown')
    vehicle_type = data.get('vehicle_type', 'Unknown')
    reason = data.get('reason', 'Under Surveillance')
    priority_str = data.get('priority', 'low')
    
    priority_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    priority = priority_map.get(str(priority_str).lower(), 1)
    
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO watchlist (plate_number, owner_name, vehicle_type, reason, priority, is_active) VALUES (?, ?, ?, ?, ?, 1)",
            (plate_number, owner_name, vehicle_type, reason, priority)
        )
        conn.commit()
        conn.close()
        return jsonify({"message": f"Vehicle {plate_number} added to watchlist"})
    except sqlite3.IntegrityError:
        cursor.execute(
            "UPDATE watchlist SET owner_name=?, vehicle_type=?, reason=?, priority=?, is_active=1 WHERE plate_number=?",
            (owner_name, vehicle_type, reason, priority, plate_number)
        )
        conn.commit()
        conn.close()
        return jsonify({"message": f"Vehicle {plate_number} updated in watchlist"})

@app.route('/api/watchlist/<plate_number>', methods=['DELETE'])
def remove_from_watchlist(plate_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE watchlist SET is_active=0 WHERE plate_number=?", (plate_number.upper(),))
    conn.commit()
    conn.close()
    return jsonify({"message": f"Vehicle {plate_number.upper()} removed from watchlist"})

@app.route('/api/anpr/detect', methods=['POST'])
def detect_plate():
    frame = None
    cam_id = 'UNKNOWN'

    if request.is_json and request.json and 'image_base64' in request.json:
        import base64
        b64 = request.json['image_base64']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cam_id = request.json.get('cam_id', 'UNKNOWN')
    elif 'frame' in request.files:
        file = request.files['frame']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cam_id = request.form.get('cam_id', 'UNKNOWN')
    else:
        return jsonify({"error": "No frame provided. Send 'frame' as file or 'image_base64' as JSON."}), 400

    if frame is None:
        return jsonify({"error": "Invalid frame data"}), 400

    h, w = frame.shape[:2]
    best_text = None
    best_conf = 0.0

    for y_pct, h_pct in [(0.5, 0.5), (0.6, 0.4), (0.0, 1.0)]:
        y1 = int(h * y_pct)
        y2 = h
        for x_pct, w_pct in [(0.0, 0.5), (0.5, 0.5), (0.0, 1.0)]:
            x1 = int(w * x_pct)
            x2 = int(w * (x_pct + w_pct))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            result = anpr_engine._engine.read_plate(frame, (x1, y1, x2, y2))
            if result and result.confidence > best_conf:
                best_conf = result.confidence
                best_text = result.plate_text

    if best_text is None:
        return jsonify({"detected": False, "message": "No plate detected in uploaded image"})

    plate_number = best_text
    confidence = best_conf
    watchlist_match = check_watchlist(plate_number)
    lat, lng = get_camera_location(cam_id)
    is_on_watchlist = watchlist_match is not None
    alert_triggered = is_on_watchlist

    log_vehicle_detection(plate_number, cam_id, confidence, is_on_watchlist, alert_triggered)
    log_movement(plate_number, cam_id, lat, lng)
    log_event(cam_id, 'anpr', plate_number, confidence)

    result = {
        "detected": True,
        "plate_number": plate_number,
        "confidence": confidence,
        "is_on_watchlist": is_on_watchlist,
        "alert_triggered": alert_triggered,
        "cam_id": cam_id
    }

    if watchlist_match:
        result['watchlist_info'] = {
            'owner_name': watchlist_match[1],
            'vehicle_type': watchlist_match[2],
            'reason': watchlist_match[3],
            'priority': watchlist_match[4]
        }

    return jsonify(result)

@app.route('/api/anpr/snap_and_analyze/<cam_id>', methods=['POST'])
def snap_and_analyze(cam_id):
    with engines_lock:
        engine = analytics_engines.get(cam_id)
    if engine is None:
        return jsonify({"error": f"No analytics engine running for {cam_id}"}), 404

    raw_frame = None
    fullres_frame = None
    with engine.lock:
        if engine.last_raw_frame is not None:
            raw_frame = engine.last_raw_frame.copy()
        if hasattr(engine, 'last_fullres_frame') and engine.last_fullres_frame is not None:
            fullres_frame = engine.last_fullres_frame.copy()
    if raw_frame is None:
        return jsonify({"error": f"No frame available for {cam_id}"}), 404

    nim_frame = fullres_frame if fullres_frame is not None else raw_frame

    h, w = raw_frame.shape[:2]
    best_text = None
    best_conf = 0.0
    best_crop = None

    for y_pct in [0.5, 0.6, 0.7, 0.0]:
        y1 = int(h * y_pct)
        y2 = h
        for x_pct in [0.0, 0.5]:
            x1 = int(w * x_pct)
            x2 = int(w * min(x_pct + 0.5, 1.0))
            crop = raw_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            result = anpr_engine._engine.read_plate(raw_frame, (x1, y1, x2, y2))
            if result and result.confidence > best_conf:
                best_conf = result.confidence
                best_text = result.plate_text
                best_crop = crop

    if best_text is None and anpr_engine.nim._ready:
        nim_plates = anpr_engine.nim.read_plates_from_frame(nim_frame)
        if nim_plates:
            best_text = nim_plates[0]
            best_conf = 0.85

    if best_text is None:
        return jsonify({
            "detected": False,
            "message": "No plate detected in current frame",
            "cam_id": cam_id,
            "frame_size": f"{nim_frame.shape[1]}x{nim_frame.shape[0]}",
            "nim_available": anpr_engine.nim._ready
        })

    plate_number = best_text
    confidence = best_conf
    watchlist_match = check_watchlist(plate_number)
    lat, lng = get_camera_location(cam_id)
    is_on_watchlist = watchlist_match is not None

    log_vehicle_detection(plate_number, cam_id, confidence, is_on_watchlist, is_on_watchlist)
    log_movement(plate_number, cam_id, lat, lng)
    log_event(cam_id, 'anpr', plate_number, confidence)

    _, buf = cv2.imencode('.jpg', best_crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) if best_crop is not None else (False, None)
    crop_b64 = None
    if buf is not None:
        import base64
        crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    resp = {
        "detected": True,
        "plate_number": plate_number,
        "confidence": confidence,
        "cam_id": cam_id,
        "is_on_watchlist": is_on_watchlist,
        "crop_base64": crop_b64
    }
    if watchlist_match:
        resp['watchlist_info'] = {
            'owner_name': watchlist_match[1],
            'vehicle_type': watchlist_match[2],
            'reason': watchlist_match[3],
            'priority': watchlist_match[4]
        }
    return jsonify(resp)

@app.route('/api/vehicles/search/<plate_number>')
def search_vehicle(plate_number):
    plate_number = plate_number.upper().replace(' ', '').replace('-', '')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    exact_count = cursor.execute(
        "SELECT COUNT(*) FROM vehicle_detections WHERE plate_number=?",
        (plate_number,)
    ).fetchone()[0]

    if exact_count > 0:
        match_type = 'exact'
        matched_plate = plate_number
    else:
        fuzzy = cursor.execute(
            "SELECT DISTINCT plate_number FROM vehicle_detections WHERE plate_number LIKE ? LIMIT 5",
            (f'%{plate_number}%',)
        ).fetchall()
        if not fuzzy:
            fuzzy = cursor.execute(
                "SELECT DISTINCT plate_number FROM vehicle_detections WHERE plate_number LIKE ? LIMIT 5",
                (f'%{plate_number[:6]}%',)
            ).fetchall()
        if fuzzy:
            matched_plate = fuzzy[0]['plate_number']
            match_type = 'fuzzy'
        else:
            conn.close()
            return jsonify({
                'plate_number': plate_number,
                'match_type': 'none',
                'is_on_watchlist': False,
                'watchlist_info': None,
                'total_detections': 0,
                'detections': [],
                'movement_history': [],
                'route': [],
                'cameras': [],
                'first_seen': None,
                'last_seen': None
            })
        plate_number = matched_plate

    detections = [dict(row) for row in cursor.execute(
        "SELECT * FROM vehicle_detections WHERE plate_number=? ORDER BY timestamp DESC LIMIT 100",
        (plate_number,)
    ).fetchall()]

    movements = [dict(row) for row in cursor.execute(
        "SELECT * FROM movement_history WHERE plate_number=? ORDER BY timestamp ASC",
        (plate_number,)
    ).fetchall()]

    watchlist = cursor.execute(
        "SELECT * FROM watchlist WHERE plate_number=? AND is_active=1",
        (plate_number,)
    ).fetchone()

    cam_ids = set()
    cam_map = {}
    for d in detections:
        cid = d['cam_id']
        cam_ids.add(cid)
        if cid not in cam_map:
            cam_map[cid] = {'count': 0}
        cam_map[cid]['count'] += 1

    cameras = []
    for cid in cam_ids:
        crow = cursor.execute(
            "SELECT name, department, latitude, longitude FROM cctv_registry WHERE cam_id=?",
            (cid,)
        ).fetchone()
        cam_info = {
            'cam_id': cid,
            'name': crow['name'] if crow else cid,
            'department': crow['department'] if crow else 'Unknown',
            'latitude': crow['latitude'] if crow else 0.0,
            'longitude': crow['longitude'] if crow else 0.0,
            'detection_count': cam_map[cid]['count']
        }
        cameras.append(cam_info)

    cameras.sort(key=lambda c: c['detection_count'], reverse=True)

    first_seen = detections[-1]['timestamp'] if detections else None
    last_seen = detections[0]['timestamp'] if detections else None

    route = []
    seen_cams = set()
    for m in movements:
        cid = m['cam_id']
        if cid in seen_cams:
            continue
        seen_cams.add(cid)
        crow = cursor.execute(
            "SELECT name, department FROM cctv_registry WHERE cam_id=?",
            (cid,)
        ).fetchone()
        route.append({
            'cam_id': cid,
            'camera_name': crow['name'] if crow else cid,
            'department': crow['department'] if crow else 'Unknown',
            'latitude': m['latitude'],
            'longitude': m['longitude'],
            'timestamp': m['timestamp'],
            'direction': m['direction'],
            'speed_kmh': m['speed_kmh']
        })

    conn.close()

    return jsonify({
        'plate_number': plate_number,
        'match_type': match_type,
        'is_on_watchlist': watchlist is not None,
        'watchlist_info': dict(watchlist) if watchlist else None,
        'total_detections': len(detections),
        'detections': detections,
        'movement_history': movements,
        'route': route,
        'cameras': cameras,
        'first_seen': first_seen,
        'last_seen': last_seen
    })

@app.route('/api/plates/suggest')
def suggest_plates():
    q = request.args.get('q', '').upper().replace(' ', '').replace('-', '')
    if len(q) < 2:
        return jsonify([])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows = cursor.execute(
        """SELECT DISTINCT plate_number, COUNT(*) as cnt
           FROM vehicle_detections
           WHERE plate_number LIKE ?
           GROUP BY plate_number
           ORDER BY cnt DESC
           LIMIT 8""",
        (f'%{q}%',)
    ).fetchall()

    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/tracking/status')
def tracking_status():
    result = {}
    for cam_id, engine in analytics_engines.items():
        if not engine.running:
            continue
        active = ANALYTICS_ACTIVE.get(cam_id, False)
        if not active:
            continue
        with engine.lock:
            tracks = []
            for tid, info in engine.track_history.items():
                positions = info['positions']
                if len(positions) < 2:
                    continue
                p1 = positions[-2]
                p2 = positions[-1]
                dt = p2[2] - p1[2]
                dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                speed = dist_px / max(dt, 0.001)
                tracks.append({
                    'track_id': tid,
                    'label': info['label'],
                    'position': [p2[0], p2[1]],
                    'speed_px_per_sec': round(speed, 1),
                    'duration_sec': round(positions[-1][2] - positions[0][2], 1),
                    'point_count': len(positions),
                    'first_seen': info['first_seen']
                })
            result[cam_id] = {
                'active_tracks': len(tracks),
                'total_tracked': engine.tracker.next_id,
                'tracks': tracks
            }
    return jsonify(result)


@app.route('/favicon.ico')
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#4f46e5"/><polygon points="50,25 70,65 30,65" fill="#ffffff"/></svg>'
    return Response(svg, mimetype='image/svg+xml')

@app.route('/api/export/<data_type>', methods=['GET'])
def export_data(data_type):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    if data_type == 'events':
        rows = cursor.execute("SELECT e.id, e.cam_id, c.name as camera_name, c.department, e.event_type, e.label, e.confidence, e.details, e.timestamp FROM events e LEFT JOIN cctv_registry c ON e.cam_id = c.cam_id ORDER BY e.timestamp DESC LIMIT 2000").fetchall()
        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(list(r))
        filename = "sentinel_events.csv"
    elif data_type == 'violations':
        rows = cursor.execute("SELECT v.*, c.name as camera_name, c.department FROM violation_evidence v LEFT JOIN cctv_registry c ON v.cam_id = c.cam_id ORDER BY v.timestamp DESC LIMIT 2000").fetchall()
        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(list(r))
        filename = "sentinel_violations.csv"
    elif data_type == 'detections':
        rows = cursor.execute("SELECT vd.id, vd.plate_number, vd.cam_id, c.name as camera_name, vd.confidence, vd.timestamp, vd.is_on_watchlist, vd.alert_triggered FROM vehicle_detections vd LEFT JOIN cctv_registry c ON vd.cam_id = c.cam_id ORDER BY vd.timestamp DESC LIMIT 2000").fetchall()
        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(list(r))
        filename = "sentinel_vehicle_detections.csv"
    elif data_type == 'watchlist':
        rows = cursor.execute("SELECT * FROM watchlist WHERE is_active=1 ORDER BY priority DESC, created_at DESC").fetchall()
        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(list(r))
        filename = "sentinel_watchlist.csv"
    elif data_type == 'anpr':
        writer.writerow(['Plate Number', 'Camera ID', 'Camera Name', 'Department', 'Latitude', 'Longitude', 'Direction', 'Speed (km/h)', 'Confidence', 'On Watchlist', 'Alert Triggered', 'Timestamp'])
        rows = cursor.execute("""
            SELECT vd.plate_number, vd.cam_id, c.name as camera_name, c.department,
                   c.latitude, c.longitude, mh.direction, mh.speed_kmh,
                   vd.confidence, vd.is_on_watchlist, vd.alert_triggered, vd.timestamp
            FROM vehicle_detections vd
            LEFT JOIN cctv_registry c ON vd.cam_id = c.cam_id
            LEFT JOIN movement_history mh ON vd.plate_number = mh.plate_number AND vd.cam_id = mh.cam_id
            ORDER BY vd.timestamp DESC LIMIT 5000
        """).fetchall()
        for r in rows:
            writer.writerow(list(r))
        filename = "sentinel_anpr_report.csv"
    else:
        conn.close()
        return jsonify({"error": f"Unknown export type '{data_type}'. Valid types: events, violations, detections, watchlist"}), 400
        
    conn.close()
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@app.route('/api/analytics/summary')
def analytics_summary():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    
    total_cams = cursor.execute("SELECT COUNT(*) FROM cctv_registry").fetchone()[0]
    online_cams = cursor.execute("SELECT COUNT(*) FROM cctv_registry WHERE status='ONLINE'").fetchone()[0]
    offline_cams = cursor.execute("SELECT COUNT(*) FROM cctv_registry WHERE status='OFFLINE'").fetchone()[0]
    degraded_cams = cursor.execute("SELECT COUNT(*) FROM cctv_registry WHERE status='DEGRADED'").fetchone()[0]
    
    total_events = cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    yolo_events = cursor.execute("SELECT COUNT(*) FROM events WHERE event_type='yolo'").fetchone()[0]
    anpr_events = cursor.execute("SELECT COUNT(*) FROM events WHERE event_type='anpr'").fetchone()[0]
    violation_events = cursor.execute("SELECT COUNT(*) FROM events WHERE event_type='violation'").fetchone()[0]
    
    total_watchlist = cursor.execute("SELECT COUNT(*) FROM watchlist WHERE is_active=1").fetchone()[0]
    total_alerts = cursor.execute("SELECT COUNT(*) FROM watchlist_alerts").fetchone()[0]
    unack_alerts = cursor.execute("SELECT COUNT(*) FROM watchlist_alerts WHERE acknowledged=0").fetchone()[0]
    
    total_vehicles = cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM vehicle_detections").fetchone()[0]
    
    conn.close()
    return jsonify({
        "cameras": {
            "total": total_cams,
            "online": online_cams,
            "offline": offline_cams,
            "degraded": degraded_cams
        },
        "events": {
            "total": total_events,
            "yolo": yolo_events,
            "anpr": anpr_events,
            "violations": violation_events
        },
        "watchlist": {
            "active_plates": total_watchlist,
            "total_alerts": total_alerts,
            "unacknowledged_alerts": unack_alerts
        },
        "vehicles": {
            "unique_tracked": total_vehicles
        }
    })

@app.route('/api/vehicles/correlate')
def correlate_vehicles():
    window_minutes = request.args.get('window', 60, type=int)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT plate_number, COUNT(DISTINCT cam_id) as cam_count, MAX(timestamp) as last_seen, MIN(timestamp) as first_seen
        FROM movement_history
        WHERE timestamp >= datetime('now', ?)
        GROUP BY plate_number
        HAVING cam_count >= 2
        ORDER BY last_seen DESC
        LIMIT 20
    """
    rows = cursor.execute(query, (f'-{window_minutes} minutes',)).fetchall()
    
    results = []
    for r in rows:
        plate = r['plate_number']
        history = cursor.execute("""
            SELECT m.*, c.name as camera_name, c.department
            FROM movement_history m
            LEFT JOIN cctv_registry c ON m.cam_id = c.cam_id
            WHERE m.plate_number = ?
            ORDER BY m.timestamp ASC
        """, (plate,)).fetchall()
        
        wl = cursor.execute("SELECT owner_name, reason, priority FROM watchlist WHERE plate_number=? AND is_active=1", (plate,)).fetchone()
        
        results.append({
            "plate_number": plate,
            "camera_count": r['cam_count'],
            "first_seen": r['first_seen'],
            "last_seen": r['last_seen'],
            "is_on_watchlist": wl is not None,
            "watchlist_info": dict(wl) if wl else None,
            "trail": [dict(h) for h in history]
        })
        
    conn.close()
    return jsonify(results)

@app.route('/api/events/stream')
def events_stream():
    def event_generator():
        last_event_id = 0
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            row = conn.execute("SELECT MAX(id) FROM events").fetchone()
            conn.close()
            if row and row[0]:
                last_event_id = row[0]
        except Exception:
            pass
            
        while True:
            time.sleep(1.5)
            try:
                conn = sqlite3.connect(DB_PATH, timeout=10.0)
                conn.row_factory = sqlite3.Row
                new_events = conn.execute(
                    "SELECT e.*, c.name as camera_name, c.department FROM events e LEFT JOIN cctv_registry c ON e.cam_id = c.cam_id WHERE e.id > ? ORDER BY e.id ASC LIMIT 10",
                    (last_event_id,)
                ).fetchall()
                
                alerts = conn.execute(
                    "SELECT a.*, c.name as camera_name FROM watchlist_alerts a LEFT JOIN cctv_registry c ON a.cam_id = c.cam_id WHERE a.acknowledged = 0 ORDER BY a.id DESC LIMIT 5"
                ).fetchall()
                conn.close()
                
                if new_events:
                    last_event_id = max(e['id'] for e in new_events)
                    payload = json.dumps({
                        "type": "new_events",
                        "events": [dict(e) for e in new_events]
                    })
                    yield f"data: {payload}\n\n"
                    
                if alerts:
                    payload = json.dumps({
                        "type": "watchlist_alerts",
                        "alerts": [dict(a) for a in alerts]
                    })
                    yield f"data: {payload}\n\n"
            except GeneratorExit:
                break
            except Exception:
                time.sleep(1.0)
                
    return Response(event_generator(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "engine": "Sentinel Sandbox v1.0"})

if __name__ == '__main__':
    mode = "DEV (local fallback)" if DEV_MODE else "PROD (live gateway required)"
    print(f"[STARTUP] Running in {mode} mode")
    app.run(host='0.0.0.0', port=5000, threaded=True)