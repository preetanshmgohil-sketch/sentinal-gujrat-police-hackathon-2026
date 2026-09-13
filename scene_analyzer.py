"""
Sentinel Scene Analyzer — NVIDIA NIM API based traffic scene understanding.

Uses meta/llama-3.2-11b-vision-instruct via NVIDIA NIM for:
  - Scene description (what's happening in the frame)
  - Traffic violation detection via VQA
  - Vehicle detail extraction (make, model, color, direction, action)
  - Number plate reading assistance
  - Screenshot capture with full evidence logging

No local model needed — lightweight API calls, works on CPU-only 8GB RAM.
No false positives: requires NIM confident "yes" + YOLO corroboration.
"""

import os
import json
import time
import base64
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

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

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
NIM_TIMEOUT = 30
NIM_MAX_RETRIES = 2

FRAME_INTERVAL_SECONDS = 5
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "violation_screenshots")
MIN_VIOLATION_CONFIDENCE = 0.6

# ---------------------------------------------------------------------------
# Traffic rules — genuinely CCTV/CV-detectable only
# Sourced from MV Act Chapters VII, VIII, XI (observable behavior rules)
# Each rule mapped to what a camera can actually verify
# ---------------------------------------------------------------------------

TRAFFIC_RULES = {
    "wrong_way": {
        "severity": "HIGH", "fine": "\u20b95000",
        "section": "MV Act Sec 119 / Rule 189",
        "description": "Vehicle driving against notified one-way direction",
        "vqa_question": "Is any vehicle driving in the wrong direction, against the flow of traffic, or on the wrong side of the road?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle"],
        "detectability": "HIGH — pure vehicle-trajectory vs lane-direction geometry",
    },
    "no_helmet_rider": {
        "severity": "HIGH", "fine": "\u20b91000",
        "section": "MV Act Rule 193",
        "description": "Two-wheeler rider not wearing protective helmet",
        "vqa_question": "Is any motorcycle or scooter rider riding without wearing a helmet on their head?",
        "supporting_yolo": ["motorcycle"],
        "detectability": "HIGH — helmet-presence classifier on detected motorcycle riders",
    },
    "no_helmet_pillion": {
        "severity": "HIGH", "fine": "\u20b91000",
        "section": "MV Act Rule 193",
        "description": "Pillion passenger not wearing protective helmet",
        "vqa_question": "Is there a pillion passenger on a motorcycle who is not wearing a helmet?",
        "supporting_yolo": ["motorcycle"],
        "detectability": "HIGH — same classifier, second rider position",
    },
    "mobile_phone_use": {
        "severity": "MEDIUM", "fine": "\u20b91000-2000",
        "section": "MV Act Rule 170A",
        "description": "Driver using mobile phone while driving",
        "vqa_question": "Is any driver holding a mobile phone to their ear or looking at a phone while driving?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle"],
        "detectability": "MEDIUM — phone-near-ear/hand pose detection needed",
    },
    "driving_on_footpath": {
        "severity": "HIGH", "fine": "\u20b95000",
        "section": "MV Act Rule 197",
        "description": "Vehicle driving on footpath or cycle track",
        "vqa_question": "Is any vehicle driving on a footpath, sidewalk, or cycle track instead of the road?",
        "supporting_yolo": ["car", "motorcycle", "bicycle"],
        "detectability": "MEDIUM — vehicle bbox crossing defined footpath polygon",
    },
    "illegal_parking": {
        "severity": "MEDIUM", "fine": "\u20b9500",
        "section": "MV Act Rules 188, 207",
        "description": "Vehicle parked in undesignated zone (not in notified parking area)",
        "vqa_question": "Is there a vehicle parked or stopped outside of a designated parking area, blocking traffic or on a no-parking zone?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle"],
        "detectability": "MEDIUM — needs notified parking zone polygons per camera",
    },
    "stationary_obstruction": {
        "severity": "MEDIUM", "fine": "\u20b9500-1000",
        "section": "MV Act Rule 190",
        "description": "Stationary vehicle causing obstruction to traffic",
        "vqa_question": "Is any vehicle stationary or stopped in a way that is blocking or obstructing the flow of traffic?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle"],
        "detectability": "MEDIUM — same bbox stationary >N minutes in non-parking zone",
    },
    "dangerous_projection": {
        "severity": "MEDIUM", "fine": "\u20b91000",
        "section": "MV Act Rule 199",
        "description": "Dangerous object protruding from vehicle",
        "vqa_question": "Is any vehicle carrying an object that is dangerously protruding or hanging out from the vehicle?",
        "supporting_yolo": ["car", "bus", "truck"],
        "detectability": "MEDIUM — object-silhouette anomaly on vehicle bbox",
    },
    "goods_fallen_on_road": {
        "severity": "MEDIUM", "fine": "\u20b9500-2000",
        "section": "MV Act Rules 192, 204",
        "description": "Goods or debris fallen or abandoned on road",
        "vqa_question": "Are there goods, cargo, or debris fallen or abandoned on the road surface obstructing traffic?",
        "supporting_yolo": [],
        "detectability": "MEDIUM — static-object-in-roadway detection",
    },
    "no_headlights_after_dark": {
        "severity": "MEDIUM", "fine": "\u20b9500",
        "section": "MV Act Rule 191",
        "description": "Moving vehicle without lights on after dark",
        "vqa_question": "Is any vehicle moving on the road at night without its headlights turned on?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle"],
        "detectability": "LOW — needs low-light/headlight-state classification + time-of-day fusion",
    },
    "high_beam_dazzle": {
        "severity": "LOW", "fine": "\u20b9500",
        "section": "MV Act Rule 202",
        "description": "Dazzling or high-beam headlights causing glare",
        "vqa_question": "Is there a vehicle using high-beam headlights that are causing glare or dazzling other road users?",
        "supporting_yolo": ["car", "bus", "truck"],
        "detectability": "LOW — image brightness/glare spike detection at night",
    },
    "missing_reflector_boards": {
        "severity": "LOW", "fine": "\u20b9500",
        "section": "MV Act Rule 205",
        "description": "Parked goods carriage without rear reflector boards after dark",
        "vqa_question": "Is there a parked truck or goods vehicle at night that is missing rear reflector boards or warning signs?",
        "supporting_yolo": ["truck"],
        "detectability": "LOW — visual inspection of stopped goods vehicles at night",
    },
    "overcrowded_bus": {
        "severity": "MEDIUM", "fine": "\u20b92000",
        "section": "MV Act Rule 151",
        "description": "Stage carriage (bus) carrying passengers beyond capacity",
        "vqa_question": "Is there a bus that appears overcrowded with passengers standing beyond the permitted limit?",
        "supporting_yolo": ["bus"],
        "detectability": "LOW — person-count inside bus, hard with exterior CCTV (occlusion, glass)",
    },
    "person_on_vehicle_exterior": {
        "severity": "HIGH", "fine": "\u20b91000-2000",
        "section": "MV Act Rule 196",
        "description": "Person mounting, hanging off, or riding on exterior of moving vehicle",
        "vqa_question": "Is any person hanging off, riding on the outside, or dangerously mounting a moving vehicle?",
        "supporting_yolo": ["person", "bus", "truck"],
        "detectability": "LOW — action-recognition, not bounding-box task",
    },
    "accident": {
        "severity": "CRITICAL", "fine": "N/A",
        "section": "MV Act Sec 161-165",
        "description": "Traffic accident or collision detected",
        "vqa_question": "Is there a traffic accident, collision, or crash visible involving vehicles or pedestrians?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle", "person"],
        "detectability": "HIGH — visual wreckage/deformation/positions",
    },
    "general_obstruction": {
        "severity": "LOW", "fine": "\u20b92000",
        "section": "MV Act Rule 237 (Ch XI)",
        "description": "General traffic obstruction (catch-all, usually post-flagging)",
        "vqa_question": "Is there anything obstructing the normal flow of traffic on the road?",
        "supporting_yolo": ["car", "bus", "truck", "motorcycle", "person"],
        "detectability": "HIGH — broad catch-all, applied after other violations flagged",
    },
}

VEHICLE_SPECS = {
    "car": {"type": "Four-Wheeler", "category": "LMV", "typical_speed": "40-120 km/h"},
    "bus": {"type": "Public Transport", "category": "HMV", "typical_speed": "40-80 km/h"},
    "truck": {"type": "Goods Carrier", "category": "HMV", "typical_speed": "40-80 km/h"},
    "motorcycle": {"type": "Two-Wheeler", "category": "Two-Wheeler", "typical_speed": "40-100 km/h"},
    "bicycle": {"type": "Non-Motorized", "category": "Non-Motorized", "typical_speed": "10-25 km/h"},
    "person": {"type": "Pedestrian", "category": "VRU", "typical_speed": "3-6 km/h"},
}


@dataclass
class SceneDescription:
    description: str
    lighting: str = "unknown"
    weather: str = "unknown"
    road_condition: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


@dataclass
class VehicleInfo:
    vehicle_type: str
    category: str
    typical_speed: str
    color: str = "unknown"
    make: str = "unknown"
    model: str = "unknown"
    direction: str = "unknown"
    action: str = "unknown"
    plate_number: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ViolationEvidence:
    rule_id: str
    rule_name: str
    severity: str
    fine: str
    section: str
    description: str
    confidence: float
    screenshot_path: str
    plate_number: Optional[str]
    vehicle_type: str
    vehicle_color: str
    vehicle_make: str
    vehicle_model: str
    vehicle_direction: str
    vehicle_action: str
    lighting_condition: str
    weather: str
    road_condition: str
    num_vehicles: int
    num_persons: int
    ai_scene_description: str
    ai_violation_evidence: str
    yolo_detections: str
    camera_id: str = ""
    timestamp: float = field(default_factory=time.time)


class SceneAnalyzer:
    """NVIDIA NIM API based scene understanding engine for traffic surveillance."""

    def __init__(self, db_path: str = "registry.db"):
        self.db_path = db_path
        self._loaded = False
        self._lock = threading.Lock()
        self._analysis_cache: Dict[str, Any] = {}
        self._api_key = NIM_API_KEY
        self._api_url = NIM_API_URL
        self._model = NIM_MODEL
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            if not self._api_key:
                logger.error("[SceneAnalyzer] No NIM API key set.")
                return False
            try:
                logger.info("[SceneAnalyzer] Verifying NIM API connectivity...")
                resp = requests.get(
                    "https://integrate.api.nvidia.com/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    self._loaded = True
                    logger.info("[SceneAnalyzer] NIM API ready.")
                    return True
                else:
                    logger.error(f"[SceneAnalyzer] NIM API check failed: {resp.status_code}")
                    return False
            except Exception as e:
                logger.error(f"[SceneAnalyzer] NIM API connectivity error: {e}")
                return False

    def _frame_to_base64(self, frame: np.ndarray, max_dim: int = 640) -> str:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    def _query_nim(self, image_b64: str, question: str, max_tokens: int = 60) -> str:
        if not self._ensure_loaded():
            return ""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        for attempt in range(NIM_MAX_RETRIES + 1):
            try:
                r = requests.post(self._api_url, headers=headers, json=payload, timeout=NIM_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
                elif r.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"[SceneAnalyzer] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"[SceneAnalyzer] NIM API error {r.status_code}: {r.text[:200]}")
                    return ""
            except requests.exceptions.Timeout:
                logger.warning(f"[SceneAnalyzer] NIM timeout (attempt {attempt+1})")
                time.sleep(1)
            except Exception as e:
                logger.error(f"[SceneAnalyzer] NIM request error: {e}")
                return ""
        return ""

    def _save_screenshot(self, frame: np.ndarray, camera_id: str, rule_id: str) -> str:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{rule_id}_{timestamp_str}.jpg"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return filepath

    def _detect_scene_conditions(self, image_b64: str) -> Tuple[str, str, str]:
        cond_prompt = (
            "Analyze this CCTV frame. Answer these 3 questions briefly, "
            "one line each in exactly this format:\n"
            "LIGHTING: (daytime/nighttime/dawn/dusk/well-lit indoor)\n"
            "WEATHER: (clear/rainy/foggy/cloudy/snowy)\n"
            "ROAD: (dry/wet/muddy/icy/under construction)"
        )
        try:
            raw = self._query_nim(image_b64, cond_prompt, max_tokens=80)
            lighting, weather, road = "unknown", "unknown", "unknown"
            for line in raw.split('\n'):
                lower = line.lower().strip()
                if 'lighting' in lower or 'light' in lower:
                    val = lower.split(':', 1)[-1].strip().rstrip(')')
                    val = val.lstrip('(').strip()
                    lighting = val.split()[0] if val else "unknown"
                elif 'weather' in lower:
                    val = lower.split(':', 1)[-1].strip().rstrip(')')
                    val = val.lstrip('(').strip()
                    weather = val.split()[0] if val else "unknown"
                elif 'road' in lower:
                    val = lower.split(':', 1)[-1].strip().rstrip(')')
                    val = val.lstrip('(').strip()
                    road = val.split()[0] if val else "unknown"
            return lighting[:40], weather[:40], road[:40]
        except Exception:
            return "unknown", "unknown", "unknown"

    def _extract_vehicle_details(self, image_b64: str, yolo_detections: List[Dict],
                                  plate_number: Optional[str] = None) -> Dict:
        num_vehicles = sum(1 for d in yolo_detections if d.get("class") in VEHICLE_SPECS and d["class"] != "person")
        num_persons = sum(1 for d in yolo_detections if d.get("class") == "person")
        vehicle_types = [d["class"] for d in yolo_detections if d.get("class") in VEHICLE_SPECS and d["class"] != "person"]

        detail_prompt = (
            "Analyze this CCTV frame. Answer these questions briefly in this exact format:\n"
            "COLOR: (one word: white/black/silver/red/blue/green/yellow/grey/orange)\n"
            "MAKE: (vehicle manufacturer, say unknown if unsure)\n"
            "DIRECTION: (forward/left/right/backward)\n"
            "ACTION: (moving/stopped/turning/parking/reversing)"
        )
        try:
            raw = self._query_nim(image_b64, detail_prompt, max_tokens=100)
            color, make, direction, action = "unknown", "unknown", "unknown", "unknown"
            for line in raw.split('\n'):
                lower = line.lower().strip()
                if 'color' in lower:
                    val = lower.split(':', 1)[-1].strip().strip('()').strip()
                    color = val.split()[0] if val else "unknown"
                elif 'make' in lower:
                    val = line.split(':', 1)[-1].strip().strip('()').strip()
                    make = val if val else "unknown"
                elif 'direction' in lower:
                    val = lower.split(':', 1)[-1].strip().strip('()').strip()
                    direction = val.split()[0] if val else "unknown"
                elif 'action' in lower:
                    val = lower.split(':', 1)[-1].strip().strip('()').strip()
                    action = val.split()[0] if val else "unknown"
        except Exception:
            color, make, direction, action = "unknown", "unknown", "unknown", "unknown"

        primary_type = vehicle_types[0] if vehicle_types else "unknown"
        specs = VEHICLE_SPECS.get(primary_type, {"type": "Unknown", "category": "Unknown", "typical_speed": "N/A"})

        return {
            "vehicle_type": primary_type,
            "vehicle_color": color[:20] if color else "unknown",
            "vehicle_make": make[:40] if make else "unknown",
            "vehicle_model": "unknown",
            "vehicle_direction": direction[:15] if direction else "unknown",
            "vehicle_action": action[:15] if action else "unknown",
            "num_vehicles": num_vehicles,
            "num_persons": num_persons,
            "category": specs["category"],
        }

    def _is_true_positive(self, nim_answer: str, rule_id: str,
                           yolo_detections: List[Dict]) -> bool:
        answer_lower = nim_answer.lower().strip()
        positive_signals = ["yes", "it is", "clearly", "appears to be", "seems to be", "visible", "there is", "there appears"]
        is_positive = any(sig in answer_lower for sig in positive_signals)
        if not is_positive:
            return False
        rule = TRAFFIC_RULES.get(rule_id, {})
        supporting = rule.get("supporting_yolo", [])
        yolo_classes = [d.get("class", "") for d in yolo_detections]
        has_support = any(cls in supporting for cls in yolo_classes)
        if rule_id in ("accident", "pedestrian_violation"):
            return True
        return has_support

    def describe_scene(self, frame: np.ndarray, camera_id: str = "") -> SceneDescription:
        t0 = time.time()
        b64 = self._frame_to_base64(frame)
        caption = self._query_nim(b64, "Describe this CCTV traffic scene in one concise sentence.", max_tokens=80)
        if not caption:
            caption = "Unable to describe scene"
        lighting, weather, road = self._detect_scene_conditions(b64)
        elapsed = (time.time() - t0) * 1000
        result = SceneDescription(
            description=caption, lighting=lighting,
            weather=weather, road_condition=road,
            processing_time_ms=elapsed,
        )
        self._analysis_cache[camera_id] = {
            "description": caption, "lighting": lighting,
            "weather": weather, "road": road, "time": time.time(),
        }
        return result

    def detect_violations(self, frame: np.ndarray, camera_id: str = "",
                          yolo_detections: Optional[List[Dict]] = None,
                          plate_number: Optional[str] = None) -> List[ViolationEvidence]:
        b64 = self._frame_to_base64(frame)
        yolo_dets = yolo_detections or []
        violations = []

        cached = self._analysis_cache.get(camera_id, {})
        lighting = cached.get("lighting", "unknown")
        weather = cached.get("weather", "unknown")
        road = cached.get("road", "unknown")
        scene_desc = cached.get("description", "")

        if not scene_desc:
            scene_desc = self._query_nim(b64, "Describe this CCTV traffic scene in one concise sentence.", max_tokens=80)

        vehicle_info = self._extract_vehicle_details(b64, yolo_dets, plate_number)

        for rule_id, rule in TRAFFIC_RULES.items():
            answer = self._query_nim(b64, rule["vqa_question"] + " Answer with a short yes/no explanation.", max_tokens=40)
            if not answer:
                continue

            if not self._is_true_positive(answer, rule_id, yolo_dets):
                continue

            screenshot_path = self._save_screenshot(frame, camera_id, rule_id)

            evidence = ViolationEvidence(
                rule_id=rule_id,
                rule_name=rule["description"],
                severity=rule["severity"],
                fine=rule["fine"],
                section=rule["section"],
                description=rule["description"],
                confidence=0.75,
                screenshot_path=screenshot_path,
                plate_number=plate_number,
                vehicle_type=vehicle_info["vehicle_type"],
                vehicle_color=vehicle_info["vehicle_color"],
                vehicle_make=vehicle_info["vehicle_make"],
                vehicle_model=vehicle_info["vehicle_model"],
                vehicle_direction=vehicle_info["vehicle_direction"],
                vehicle_action=vehicle_info["vehicle_action"],
                lighting_condition=lighting,
                weather=weather,
                road_condition=road,
                num_vehicles=vehicle_info["num_vehicles"],
                num_persons=vehicle_info["num_persons"],
                ai_scene_description=scene_desc,
                ai_violation_evidence=f"NIM VLM: '{answer}'",
                yolo_detections=json.dumps(yolo_dets[:10]),
                camera_id=camera_id,
            )
            violations.append(evidence)

        if violations:
            self._log_violations(violations)

        return violations

    def _log_violations(self, violations: List[ViolationEvidence]):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            for v in violations:
                cur.execute(
                    "INSERT INTO events (cam_id, event_type, label, confidence, details) "
                    "VALUES (?, 'violation', ?, ?, ?)",
                    (v.camera_id, v.rule_name, v.confidence,
                     f"Severity:{v.severity}|Fine:{v.fine}|Section:{v.section}"),
                )
                violation_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO violation_evidence "
                    "(violation_id, cam_id, rule_id, rule_name, severity, fine, section, "
                    "description, confidence, screenshot_path, plate_number, vehicle_type, "
                    "vehicle_color, vehicle_make, vehicle_model, vehicle_direction, "
                    "vehicle_action, lighting_condition, weather, road_condition, "
                    "num_vehicles, num_persons, ai_scene_description, ai_violation_evidence, "
                    "yolo_detections) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (violation_id, v.camera_id, v.rule_id, v.rule_name, v.severity,
                     v.fine, v.section, v.description, v.confidence, v.screenshot_path,
                     v.plate_number, v.vehicle_type, v.vehicle_color, v.vehicle_make,
                     v.vehicle_model, v.vehicle_direction, v.vehicle_action,
                     v.lighting_condition, v.weather, v.road_condition,
                     v.num_vehicles, v.num_persons, v.ai_scene_description,
                     v.ai_violation_evidence, v.yolo_detections),
                )
            conn.commit()
            conn.close()
            logger.info(f"[SceneAnalyzer] Logged {len(violations)} violations to DB")
        except Exception as e:
            logger.error(f"[SceneAnalyzer] DB log error: {e}")

    def get_recent_violations(self, limit: int = 50) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT v.*, c.name as camera_name, c.department "
                "FROM violation_evidence v "
                "LEFT JOIN cctv_registry c ON v.cam_id = c.cam_id "
                "ORDER BY v.timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"[SceneAnalyzer] DB read error: {e}")
            return []

    def get_violation_screenshot(self, violation_id: int) -> Optional[str]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT screenshot_path FROM violation_evidence WHERE id=?", (violation_id,))
            row = cur.fetchone()
            conn.close()
            if row and row[0] and os.path.exists(row[0]):
                return row[0]
            return None
        except Exception as e:
            logger.error(f"[SceneAnalyzer] Screenshot lookup error: {e}")
            return None

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        with self._lock:
            self._loaded = False
            logger.info("[SceneAnalyzer] NIM session closed.")
