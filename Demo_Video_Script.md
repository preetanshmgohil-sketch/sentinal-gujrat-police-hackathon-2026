# Sentinel VMS — Demo Video Script
## Gujarat Police Innovation Hackathon 2026

**Duration:** 2-3 minutes max
**Recording:** Screen-recorded (OBS / Windows Game Bar)

---

## SCENE 1: Title (5 seconds)
- Show browser opening `http://localhost:5000`
- Dashboard loads with map and video wall
- **Voiceover:** "Sentinel VMS — a unified CCTV integration and analytics platform for Gujarat Police"

## SCENE 2: GIS Map + Registry (15 seconds)
- Click "View All Cameras" on map
- Zoom into Ahmedabad area — show 30 camera markers
- Click a marker → popup shows camera details (name, department, GPS, status)
- **Voiceover:** "30 government cameras from multiple departments are registered with metadata — location, department, connectivity status"

## SCENE 3: Video Wall — Camera Feed (20 seconds)
- Click a camera marker → feeds into Slot 1 of the 2x2 grid
- Video starts playing (Progressive MP4)
- Camera name, department, and LIVE badge visible
- Click a second camera → Slot 2
- **Voiceover:** "Live feeds load directly into a 2x2 video wall — no middleware, no extra infrastructure"

## SCENE 4: AI Analytics — Object Detection (20 seconds)
- Click "AI: OFF" on Slot 1 → toggles to "AI: ON"
- Detection feed panel appears at bottom
- Show detections appearing: "person (87%)", "motorcycle (72%)", "car (91%)"
- **Voiceover:** "Toggle AI on any camera — YOLOv8 Nano detects persons, cars, motorcycles, buses, and trucks in real-time"

## SCENE 5: ANPR — Plate Recognition (20 seconds)
- Camera with vehicles loads in grid
- Show plate detection appearing in detection feed
- Example: "Plate: GJ06DB4378 (97%)"
- Navigate to ANPR Alerts page → show the plate entry with timestamp, camera name
- **Voiceover:** "ANPR reads Indian number plates with 97% accuracy — EasyOCR with multi-scale detection"

## SCENE 6: Watchlist Matching (15 seconds)
- Show watchlist page with a plate added (e.g., stolen vehicle)
- Detection feed shows "WATCHLIST HIT" alert
- Alert flash bar appears at top of dashboard
- **Voiceover:** "Cross-references every plate against the watchlist — instant alerts for stolen vehicles, wanted persons"

## SCENE 7: Vehicle Tracking (20 seconds)
- Navigate to Vehicle Search page
- Enter a plate number (e.g., "GJ06DB4378")
- Show route map with camera stops
- Show movement timeline with timestamps, directions, speeds
- **Voiceover:** "Track any vehicle across the CCTV network — route reconstruction with direction, speed, and camera frequency"

## SCENE 8: Events Log (10 seconds)
- Navigate to Events page
- Show filterable table with 50+ events
- Filter by YOLO/ANPR, by camera
- **Voiceover:** "Every detection is logged and searchable — filter by type, camera, timestamp"

## SCENE 9: Closing (5 seconds)
- Back to dashboard showing all features
- **Voiceover:** "Sentinel VMS — scalable, efficient, and ready for 80,000 cameras across Gujarat"

---

## Recording Tips
- Use **OBS Studio** (free) or **Windows + G** for screen recording
- Record at **1920×1080** if possible, **1280×720** minimum
- Keep mouse movements smooth and deliberate
- Pause 1-2 seconds between actions for viewer comprehension
- Record voiceover separately and sync in editing (or use text overlays)

## Preparation Checklist
- [ ] Start Flask server: `python app.py`
- [ ] Open browser at `http://localhost:5000`
- [ ] Load 2 cameras into grid before recording
- [ ] Add a test plate to watchlist (e.g., "GJ06DB4378")
- [ ] Pre-record detection data (analytics runs on grid cameras)
- [ ] Test all pages load correctly
- [ ] Clear browser cache for clean recording
