# Sentinel VMS — Unified CCTV Integration Platform

> Gujarat Police Innovation Hackathon 2026 — Model 1 (Registry & GIS) + Model 2 (Unified Viewing & Analytics)

Unifies 26 departments' independent CCTV systems into a single viewing, GIS mapping, and AI analytics platform. Connects to **30 live cameras** across Gujarat (Junagadh, Ahmedabad, Rajkot, Navsari, Gandhinagar, Gandhidham, Patan, Kheda).

---

## Features

| Module | What it does |
|--------|-------------|
| **Camera Registry** | 30 cameras with department, city, status, GPS coordinates |
| **GIS Map** | Interactive Leaflet map with camera markers across Gujarat |
| **Video Wall** | 2x2 grid with live RTSP streaming, AI toggle per tile |
| **YOLO Analytics** | Real-time object detection (cars, trucks, buses, persons, motorcycles) |
| **Vehicle Tracking** | Centroid-based multi-object tracker with unique IDs, speed, duration |
| **Event Log** | Searchable detection history with camera, class, confidence, timestamp |
| **Watchlist** | Add plates with priority levels (low/medium/high/critical), auto-alerts |
| **ANPR Alerts** | Number plate detection page with CSV export |
| **Vehicle Search** | Search any plate to view detection history + movement route on map |
| **Scene Analysis** | NVIDIA NIM vision model for violation detection |
| **Snap & Analyze** | Capture a frame and send to AI for analysis |

## Screenshots

Open `http://localhost:5000` after running to see:
- Interactive Gujarat map with 30 camera markers
- Live 2x2 video grid with AI overlay (bounding boxes, tracking IDs)
- Detection panel, event log, watchlist management

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| Database | SQLite (WAL mode) |
| AI Detection | YOLOv8 Nano (Ultralytics) |
| Tracking | Centroid-based multi-object tracker |
| OCR | EasyOCR |
| Vision AI | NVIDIA NIM (Llama 3.2 11B Vision) |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Map | Leaflet.js + OpenStreetMap |
| Video | RTSP via FFmpeg + OpenCV, HLS.js fallback |
| Streaming | MJPEG over HTTP |

## Quick Start

### Prerequisites — Install These First

#### 1. Install Python 3.10+

**Windows:**
- Download from https://www.python.org/downloads/
- Run the installer
- **IMPORTANT:** Check the box that says **"Add python.exe to PATH"** during installation
- Verify installation: open Command Prompt and run:
```
python --version
```
You should see `Python 3.10.x` or higher.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

**macOS:**
```bash
brew install python@3.10
python3 --version
```

#### 2. Install Git

**Windows:**
- Download from https://git-scm.com/download/win
- Run the installer with default settings
- Verify: open Command Prompt and run:
```
git --version
```

**Linux:**
```bash
sudo apt install git
```

**macOS:**
```bash
xcode-select --install
```

#### 3. Install FFmpeg (required for RTSP video streaming)

**Windows:**
1. Download from https://www.gyan.dev/ffmpeg/builds/ — get the **"ffmpeg-release-essentials.zip"**
2. Extract the zip file (e.g. to `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Press `Win + S`, search **"Environment Variables"**
   - Click **"Environment Variables"**
   - Under **"System variables"**, find `Path`, click **Edit**
   - Click **New**, paste `C:\ffmpeg\bin`
   - Click **OK** on all dialogs
4. Verify: open a **new** Command Prompt and run:
```
ffmpeg -version
```
You should see ffmpeg version info.

**Linux:**
```bash
sudo apt install ffmpeg
ffmpeg -version
```

**macOS:**
```bash
brew install ffmpeg
ffmpeg -version
```

#### 4. Install Microsoft Visual C++ Build Tools (Windows only, for OpenCV)

If `pip install` fails with errors about `cv2` or `opencv`:
1. Download **Build Tools for Visual Studio** from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select **"Desktop development with C++"** workload
4. Click Install (requires ~3GB disk space)
5. Restart your computer after installation

---

### Setup — Follow Each Step

#### Step 1: Clone the Repository

Open Command Prompt (Windows) or Terminal (Linux/Mac) and run:

```bash
git clone https://github.com/YOUR_USERNAME/sentinal.git
cd sentinal
```

#### Step 2: Create a Virtual Environment

This creates an isolated Python environment so packages don't conflict with other projects.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your command prompt should show `(venv)` at the start.

#### Step 3: Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `flask` — web server
- `opencv-python-headless` — video processing and computer vision
- `numpy` — numerical arrays
- `requests` — HTTP client for gateway
- `ultralytics` — YOLOv8 AI model (~200MB download on first install)
- `easyocr` — text recognition for license plates (~500MB download on first run, includes PyTorch)
- `python-dotenv` — environment variable loading

**If OpenCV fails to install**, make sure you have Microsoft Visual C++ Build Tools (Windows) or `build-essential` (Linux):
```bash
# Linux only:
sudo apt install build-essential python3-dev
```

**If EasyOCR fails**, you may need to install PyTorch manually first:
```bash
# CPU only (recommended for 8GB RAM systems):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install easyocr
```

#### Step 4: Download AI Models (Automatic)

Models are downloaded **automatically on first run**:
- `yolov8n.pt` (~6MB) — YOLOv8 Nano for object detection, downloaded by Ultralytics
- `models/best.pt` (~5MB) — License plate detection model, included in repo

EasyOCR will also download its recognition models on first use (~100MB, cached in `~/.EasyOCR/`).

#### Step 5: Configure Environment (Optional)

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Open `.env` in any text editor. The defaults work out of the box. Only change if you need:

| Variable | What to set |
|----------|-------------|
| `SENTINEL_PORT` | Change `5000` to another port if 5000 is in use |
| `NVIDIA_NIM_API_KEY` | Get from https://build.nvidia.com (optional, for scene analysis) |

#### Step 6: Run the Application

```bash
python app.py
```

On first run it will:
1. Create `registry.db` SQLite database
2. Fetch 30 cameras from Gujarat Police Sentinel gateway
3. Connect to live RTSP feeds
4. Start YOLO detection on enabled cameras

Open your browser and go to: **http://localhost:5000**

#### Windows One-Click Launch

After initial setup, double-click `start.cmd` to launch without typing commands.

---

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not recognized | Python not in PATH. Reinstall Python and check "Add to PATH" |
| `ffmpeg` not recognized | Add `C:\ffmpeg\bin` to PATH, restart terminal |
| `pip install` fails on opencv | Install Microsoft Visual C++ Build Tools (see above) |
| Port 5000 in use | Change `SENTINEL_PORT=5080` in `.env` |
| Cameras show "Connecting..." | Gateway may be down. Check https://live.sentinelgujarat.in |
| EasyOCR slow first run | Normal — downloading ~100MB model. Subsequent runs are fast |
| `No module named 'distutils'` | Run `pip install setuptools` |
| Memory error on 8GB RAM | Close other applications. System uses ~3-4GB with 4 cameras active |

## Project Structure

```
sentinal/
├── app.py                  # Main Flask app — routes, engines, analytics, APIs
├── anpr.py                 # ANPR module — EasyOCR backend, plate validation
├── scene_analyzer.py       # NVIDIA NIM vision model integration
├── auth_gateway.py         # Sentinel gateway authentication
├── requirements.txt        # Python dependencies
├── .env.example            # Environment config template
├── start.cmd               # Windows one-click launcher
│
├── models/
│   └── best.pt             # YOLOv8 license plate detection model
│
├── templates/
│   ├── index.html          # Main dashboard — map + video wall + panels
│   ├── events.html         # Searchable event log
│   ├── watchlist.html      # Watchlist management
│   ├── anpr_alerts.html    # ANPR alerts with CSV export
│   ├── vehicle_search.html # Vehicle search + movement route map
│   └── violations.html     # Violation evidence viewer
│
├── static/
│   └── spa.js              # Client-side SPA router
│
├── assets/                 # Local test video files
├── registry.db             # SQLite database (auto-created on first run)
└── yolov8n.pt              # YOLOv8 Nano model (~6MB)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cameras` | GET | List all cameras with metadata |
| `/api/cameras/<id>` | GET | Single camera details |
| `/api/toggle_analytics` | POST | Toggle YOLO detection on a camera |
| `/api/analytics/status` | GET | Active cameras, detection stats |
| `/api/analytics/detections` | GET | Recent detection log |
| `/api/tracking/status` | GET | Active vehicle tracks with speed/position |
| `/api/watchlist` | GET/POST | List or add watchlist entries |
| `/api/watchlist/<id>` | DELETE | Remove watchlist entry |
| `/api/vehicles/search/<plate>` | GET | Search vehicle by plate number |
| `/api/events` | GET | Searchable event log with filters |
| `/api/export/csv` | GET | Export events as CSV |
| `/api/export/pdf` | GET | Export events as PDF |
| `/api/anpr/snap_and_analyze/<id>` | POST | Capture frame + AI analysis |
| `/video_feed/<id>` | GET | MJPEG stream for a camera |
| `/analytics_feed/<id>` | GET | MJPEG stream with YOLO overlay |
| `/hls_feed/<id>` | GET | HLS stream for a camera |

## Architecture

```
Browser (HLS.js / MJPEG)     Flask Server               Gujarat CCTV Gateway
         │                         │                            │
         │── GET /api/cameras ────>│── GET /api/ingest ───────>│ live.sentinelgujarat.in
         │<── JSON 30 cameras ────│<── JSON catalogue ────────│
         │                         │                            │
         │── GET /video_feed ────>│── VideoCapture(RTSP) ────>│ /stream/<id>
         │<── MJPEG frames ───────│── YOLO predict ──> frame   │
         │                         │── CentroidTracker ────────│
         │                         │── imencode JPEG ─────────>│
```

- **AnalyticsEngine** — Background thread per camera, connects via RTSP/TCP, runs YOLO on frames, tracks vehicles
- **SentinelStreamEngine** — Background thread per camera, serves MJPEG for video wall
- **ANPREngine** — EasyOCR-based plate detection (requires ANPR-grade cameras for reliable reads)
- **CentroidTracker** — Multi-object tracker assigning unique IDs across frames

## Camera Feeds

Connects to Gujarat Police Sentinel gateway at `https://live.sentinelgujarat.in`. Returns 30 live cameras with RTSP, HLS, and WebRTC URLs across:

| City | Cameras |
|------|---------|
| Junagadh | 10 |
| Ahmedabad | 5 |
| Navsari | 5 |
| Gandhinagar | 3 |
| Rajkot | 2 |
| Kheda | 2 |
| Gandhidham | 1 |
| Patan | 1 |
| Morbi | 1 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINEL_HOST` | `0.0.0.0` | Server bind address |
| `SENTINEL_PORT` | `5000` | Server port |
| `SENTINEL_GATEWAY` | `https://live.sentinelgujarat.in` | CCTV gateway URL |
| `NVIDIA_NIM_API_KEY` | *(empty)* | NVIDIA NIM API key for scene analysis |
| `NIM_MODEL` | `meta/llama-3.2-11b-vision-instruct` | Vision model |
| `ANPR_BACKEND` | `easyocr` | OCR backend |
| `ANPR_MIN_CONFIDENCE` | `0.25` | Minimum confidence threshold |

## License

Built for **Gujarat Police Innovation Hackathon 2026**.
