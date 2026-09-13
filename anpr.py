"""
anpr.py — Automatic Number Plate Recognition for Sentinel VMS
================================================================

Drop-in module for the Sentinel VMS AnalyticsEngine. Given a vehicle
bounding-box crop from a YOLO frame, this module:

  1. Localizes candidate plate regions inside the crop (contour-based,
     falls back to whole-crop if no candidate found — real CCTV footage
     is often too low-res/angled for clean plate segmentation, and a
     failed localization should degrade gracefully, not return nothing).
  2. Runs OCR at multiple preprocessing variants and widths, since a
     single fixed pipeline is brittle against real-world lighting/angle.
  3. Validates + normalizes the result against Indian plate formats.
  4. Applies temporal dedup and writes to the DB schema from the HLD
     (vehicle_detections, watchlist, watchlist_alerts).

HONESTY NOTE FOR YOUR SUBMISSION:
This has been tested against synthetic (rendered) plate images in this
environment — it correctly reads clean, well-lit synthetic text. It has
NOT been tested against real CCTV footage, which is much harder: motion
blur, oblique viewing angles, low resolution, night-time IR glare, and
partial occlusion all degrade OCR accuracy significantly. Before you
claim "ANPR: Complete/verified" in your HLD, run this against actual
frames from your camera grid and report the real hit rate — don't
present synthetic-test success as field-verified accuracy.

Dependencies: opencv-python, pytesseract (+ system tesseract-ocr binary).
Optional:     easyocr (better accuracy, heavier — see OCRBackend below).
"""

from __future__ import annotations

import re
import sqlite3
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sentinel.anpr")

# ---------------------------------------------------------------------------
# Indian plate format validation
# ---------------------------------------------------------------------------
# Standard format: [State(2 letters)][RTO code(1-2 digits)][Series(0-3 letters)][Number(4 digits)]
#   e.g. GJ01AB1234, MH12AB5678, DL3CAB1234
STANDARD_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

# BH-series (Bharat series, newer format): [Year(2 digits)]BH[Number(4 digits)][Series(1-2 letters)]
#   e.g. 23BH1234AB
BH_SERIES_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Characters OCR commonly confuses on Indian plates (fixed-width fonts, glare).
# Used ONLY as fallback candidates if the raw read fails validation — the raw
# read is always tried first so we never "correct" a plate into a different
# real one.
_OCR_CONFUSIONS = {
    "O": "0", "0": "O",
    "I": "1", "1": "I",
    "S": "5", "5": "S",
    "B": "8", "8": "B",
    "Z": "2", "2": "Z",
}


def normalize_plate_text(raw: str) -> str:
    """Strip whitespace/punctuation, uppercase. Does not validate format."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def validate_plate(text: str) -> bool:
    return bool(STANDARD_PLATE_RE.match(text) or BH_SERIES_RE.match(text))


def best_effort_correct(text: str) -> Optional[Tuple[str, bool]]:
    """
    If `text` validates as-is, returns (text, False) — no correction needed.
    Otherwise tries single-character substitutions at each position using
    the confusion table and returns the first validating candidate as
    (candidate, True) to mark it as a GUESS, not a clean read.

    IMPORTANT: a "corrected" result is not necessarily the right plate —
    it's the first format-valid guess. Multiple corrections can validate
    for the same misread (e.g. GJ01AB12S4 could validly correct to
    GJ01AB1254 OR the true original could be GJ01AB1234 — both pass the
    regex). Callers MUST treat corrected results as lower-trust: never
    auto-fire a high-priority watchlist alert on a corrected read alone.
    """
    if validate_plate(text):
        return text, False
    for i, ch in enumerate(text):
        if ch in _OCR_CONFUSIONS:
            candidate = text[:i] + _OCR_CONFUSIONS[ch] + text[i + 1:]
            if validate_plate(candidate):
                return candidate, True
    return None


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

class OCRBackend:
    """Base interface. Swap implementations without touching the pipeline."""

    def read_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Return list of (text, confidence 0-1) candidates, best first."""
        raise NotImplementedError


class TesseractBackend(OCRBackend):
    """Default backend. No GPU, no large model download, works out of the box
    if `tesseract-ocr` is installed on the host (apt install tesseract-ocr)."""

    def __init__(self):
        import pytesseract  # raises ImportError with a clear message if missing
        self._pytesseract = pytesseract
        # PSM 7 = "treat the image as a single text line", the correct mode
        # for a plate-shaped crop. PSM 8 ("single word") was tested and
        # found to return CORRECT text with near-zero confidence on plate
        # crops — it silently broke confidence-based filtering. Don't
        # revert to PSM 8 without re-testing against real plate crops.
        self._config = (
            "--psm 7 --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    def read_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        data = self._pytesseract.image_to_data(
            image, config=self._config,
            output_type=self._pytesseract.Output.DICT,
        )
        results = []
        for text, conf in zip(data["text"], data["conf"]):
            text = normalize_plate_text(text)
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = -1.0
            if text and conf_f >= 0:
                results.append((text, conf_f / 100.0))
        results.sort(key=lambda r: r[1], reverse=True)
        return results


class EasyOCRBackend(OCRBackend):
    """Optional, higher-accuracy backend. Not installed by default — this
    class only becomes usable if you `pip install easyocr` yourself. Heavier
    (pulls in a PyTorch detection+recognition model, first call downloads
    weights), but generally more robust to angle/lighting than Tesseract."""

    _reader = None  # lazy singleton — model load is expensive

    def __init__(self):
        import easyocr  # raises ImportError with a clear message if missing
        if EasyOCRBackend._reader is None:
            logger.info("[ANPR] Loading EasyOCR model (first call only)...")
            EasyOCRBackend._reader = easyocr.Reader(["en"], gpu=False)
        self._reader = EasyOCRBackend._reader

    def read_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        raw = self._reader.readtext(image, detail=1)
        results = [(normalize_plate_text(text), float(conf)) for (_, text, conf) in raw]
        results = [(t, c) for (t, c) in results if t]
        results.sort(key=lambda r: r[1], reverse=True)
        return results


def get_backend(name: str = "tesseract") -> OCRBackend:
    if name == "tesseract":
        return TesseractBackend()
    if name == "easyocr":
        return EasyOCRBackend()
    raise ValueError(f"Unknown OCR backend: {name}")


# ---------------------------------------------------------------------------
# Plate localization + preprocessing
# ---------------------------------------------------------------------------

def _candidate_plate_regions(vehicle_crop: np.ndarray) -> List[np.ndarray]:
    """
    Try to find plate-like rectangular regions inside a vehicle crop using
    multiple detection strategies. Falls back to lower half if nothing found.
    """
    if vehicle_crop.size == 0:
        return []

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    candidates = []

    # Enhance contrast first — helps find plates in dark/low-contrast images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Method 1: Canny edges + contours
    blurred = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / (w * h)
        if 1.8 <= aspect <= 6.0 and 0.015 <= area_frac <= 0.7:
            candidates.append(gray[y:y + ch, x:x + cw])

    # Method 2: adaptive threshold
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 9)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:15]
    for c in contours2:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / (w * h)
        if 1.5 <= aspect <= 6.5 and 0.015 <= area_frac <= 0.7:
            candidates.append(gray[y:y + ch, x:x + cw])

    # Method 3: morphological close — connects broken plate characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours3, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3 = sorted(contours3, key=cv2.contourArea, reverse=True)[:10]
    for c in contours3:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / (w * h)
        if 1.5 <= aspect <= 6.5 and 0.015 <= area_frac <= 0.7:
            candidates.append(gray[y:y + ch, x:x + cw])

    # Method 4: Otsu threshold + findContours
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours4, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours4 = sorted(contours4, key=cv2.contourArea, reverse=True)[:10]
    for c in contours4:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / (w * h)
        if 1.5 <= aspect <= 6.5 and 0.015 <= area_frac <= 0.7:
            candidates.append(gray[y:y + ch, x:x + cw])

    # Method 5: lower half fallback (typical plate position)
    candidates.append(gray[int(h * 0.55):h, :])

    return candidates


def _preprocess_for_ocr(region: np.ndarray, target_width: int) -> np.ndarray:
    """Resize + enhance a candidate region for OCR. Returns multiple variants."""
    if region.size == 0:
        return region
    h, w = region.shape[:2]
    if w == 0:
        return region
    scale = target_width / w
    resized = cv2.resize(region, (target_width, max(1, int(h * scale))),
                          interpolation=cv2.INTER_CUBIC)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Sharpen
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # Otsu threshold
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

@dataclass
class PlateResult:
    plate_text: str
    confidence: float
    was_corrected: bool = False
    raw_candidates: List[str] = field(default_factory=list)


class ANPREngine:
    def __init__(self, backend: str = "tesseract", min_confidence: float = 0.25):
        self.backend = get_backend(backend)
        self.min_confidence = min_confidence
        self._widths = (200, 300, 400, 500)

    def read_plate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """
        frame: full camera frame (BGR, as decoded from the MJPEG/RTSP feed)
        bbox:  (x1, y1, x2, y2) of a YOLO 'car'/'motorcycle'/'bus'/'truck' detection
        """
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return None

        best: Optional[Tuple[str, float, bool]] = None
        all_candidates: List[str] = []

        for region in _candidate_plate_regions(vehicle_crop):
            for width in self._widths:
                processed = _preprocess_for_ocr(region, width)
                if processed.size == 0:
                    continue
                try:
                    ocr_results = self.backend.read_text(processed)
                except Exception as e:
                    logger.warning(f"[ANPR] OCR backend error: {e}")
                    continue

                for text, conf in ocr_results:
                    all_candidates.append(text)
                    if len(text) < 6 or len(text) > 11:
                        continue
                    correction = best_effort_correct(text)
                    if correction is None:
                        continue
                    corrected_text, was_corrected = correction
                    effective_conf = conf * (0.6 if was_corrected else 1.0)
                    if best is None or effective_conf > best[1]:
                        best = (corrected_text, effective_conf, was_corrected)

            if best is not None and best[1] >= 0.85:
                break

        if best is None or best[1] < self.min_confidence:
            return None

        plate_text, conf, was_corrected = best
        if was_corrected and conf < max(self.min_confidence * 2, 0.5):
            logger.info(f"[ANPR] Discarding low-confidence corrected guess: {plate_text} ({conf:.2f})")
            return None

        return PlateResult(
            plate_text=plate_text, confidence=conf,
            was_corrected=was_corrected, raw_candidates=all_candidates,
        )


# ---------------------------------------------------------------------------
# DB integration — matches the schema in the Sentinel VMS HLD §4.3
# ---------------------------------------------------------------------------

DEDUP_WINDOW_SECONDS = 60


def log_vehicle_detection(db_path: str, plate_text: str, cam_id: str, confidence: float) -> bool:
    """
    Insert or update a vehicle_detections row, applying the 60s temporal
    dedup rule from the HLD (same plate + same camera within the window
    updates the existing row instead of inserting a duplicate).

    Returns True if this call resulted in a NEW detection (i.e. should be
    checked against the watchlist / surfaced as a fresh event), False if it
    was a dedup update of an existing recent detection.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp FROM vehicle_detections
            WHERE plate_number = ? AND cam_id = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (plate_text, cam_id),
        )
        row = cur.fetchone()

        if row is not None:
            existing_id, existing_ts = row
            cur.execute("SELECT (julianday('now') - julianday(?)) * 86400.0", (existing_ts,))
            age_seconds = cur.fetchone()[0]
            if age_seconds is not None and age_seconds < DEDUP_WINDOW_SECONDS:
                cur.execute(
                    "UPDATE vehicle_detections SET confidence = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (confidence, existing_id),
                )
                conn.commit()
                return False

        cur.execute(
            """
            INSERT INTO vehicle_detections (plate_number, cam_id, confidence, is_on_watchlist)
            VALUES (?, ?, ?, 0)
            """,
            (plate_text, cam_id, confidence),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def check_watchlist_and_alert(
    db_path: str, plate_text: str, cam_id: str, confidence: float, was_corrected: bool = False
) -> Optional[dict]:
    """
    Checks plate_text against the watchlist table (exact match first, then
    LIKE fallback per the HLD's fuzzy-search note). If matched, writes a
    watchlist_alerts row and marks the vehicle_detections row as a hit.
    Returns the alert dict if a match was found, else None.

    SAFETY: if `was_corrected` is True (the plate text came from a guessed
    single-character OCR correction rather than a clean read — see
    anpr.best_effort_correct), this refuses to fire an alert below 0.6
    confidence. A guessed correction matching the watchlist by coincidence
    of format is exactly the kind of false positive a real police alert
    pipeline cannot afford. Exact-match hits on a *clean* read always
    proceed regardless of this stricter bar.
    """
    if was_corrected and confidence < 0.6:
        logger.info(
            f"[ANPR] Skipping watchlist check for low-confidence corrected "
            f"guess '{plate_text}' (conf={confidence:.2f}) — not clean-read."
        )
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM watchlist WHERE plate_number = ?", (plate_text,))
        match = cur.fetchone()

        if match is None:
            # Fuzzy fallback — handles minor OCR variance not caught by
            # best_effort_correct (e.g. a swapped digit outside our
            # single-substitution confusion table).
            cur.execute(
                "SELECT * FROM watchlist WHERE plate_number LIKE ?",
                (f"%{plate_text[2:-2]}%",) if len(plate_text) > 4 else (plate_text,),
            )
            match = cur.fetchone()

        if match is None:
            return None

        cur.execute(
            """
            INSERT INTO watchlist_alerts
                (plate_number, cam_id, owner_name, reason, priority, confidence, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (plate_text, cam_id, match["owner_name"], match["reason"], match["priority"], confidence),
        )
        cur.execute(
            "UPDATE vehicle_detections SET is_on_watchlist = 1 WHERE plate_number = ? AND cam_id = ?",
            (plate_text, cam_id),
        )
        conn.commit()

        return {
            "plate_number": plate_text,
            "cam_id": cam_id,
            "owner_name": match["owner_name"],
            "reason": match["reason"],
            "priority": match["priority"],
            "confidence": confidence,
        }
    finally:
        conn.close()
