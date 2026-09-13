# Sentinel VMS — High-Level Design Document
## Gujarat Police Innovation Hackathon 2026
### Integrated Video Management & Analytics Platform

---

## 1. Executive Summary

**Sentinel VMS** is a unified CCTV integration and analytics platform designed to bridge 26 independent departmental CCTV systems across Gujarat into a single, intelligent monitoring ecosystem. Our solution combines **Model 1 (Centralised CCTV Registry & GIS Mapping)** and **Model 2 (Unified Viewing & Metadata Analytics)** to deliver a cost-effective, scalable platform that respects existing infrastructure while enabling AI-powered real-time analytics.

**Key Innovation:** Zero-middleware architecture — the platform connects directly to departmental camera streams via standard protocols (RTSP, HLS, ONVIF) without requiring changes to existing VMS systems, while sharing a single connection between streaming and analytics engines to minimise bandwidth and CPU overhead.

---

## 2. Problem Context

| Dimension | Challenge |
|-----------|-----------|
| Scale | 26 departments, ~80,000 cameras statewide |
| Heterogeneity | Analog + IP cameras, multiple VMS vendors, 7–15 day retention |
| Geography | 1,000 km spread — Valsad to Dwarka |
| Integration | Independent systems with no central inventory or coordination |
| Intelligence | No unified analytics, watchlist cross-referencing, or vehicle tracking |

**Our Focus:** Build a working prototype demonstrating registry + GIS mapping + unified viewing + AI analytics (YOLOv8 + ANPR) + watchlist alerting + vehicle tracking — all running on commodity hardware (8GB RAM, CPU-only).

---

## 3. Architecture Overview

### 3.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SENTINEL VMS PLATFORM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Camera   │  │  Camera   │  │  Camera   │  │  Camera   │   │
│  │  Grid 1   │  │  Grid 2   │  │  Grid 3   │  │  Grid 4   │   │
│  │  (Slot 0) │  │  (Slot 1) │  │  (Slot 2) │  │  (Slot 3) │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │              │          │
│        └──────────────┴──────┬───────┴──────────────┘          │
│                              │                                 │
│                    ┌─────────▼─────────┐                      │
│                    │   SentinelStream   │                      │
│                    │      Engine        │                      │
│                    │  (HLS → JPEG)      │                      │
│                    └─────────┬─────────┘                      │
│                              │ frame_bytes (shared)            │
│                    ┌─────────▼─────────┐                      │
│                    │  AnalyticsEngine   │                      │
│                    │  YOLOv8 Nano + ANPR│                      │
│                    └─────────┬─────────┘                      │
│                              │                                 │
│        ┌─────────────────────┼─────────────────────┐          │
│        │                     │                     │          │
│  ┌─────▼─────┐  ┌───────────▼───────────┐  ┌──────▼──────┐  │
│  │  SQLite   │  │   Event/Alert Engine   │  │  GIS Map    │  │
│  │  Database │  │   Watchlist Matching   │  │  (Leaflet)  │  │
│  └───────────┘  └───────────────────────┘  └─────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Web UI (Flask + Tailwind)                  │
│  Dashboard │ Video Wall │ Events │ ANPR Alerts │ Vehicle Search│
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Interaction Flow

```
User Action                    Backend Response
─────────────                  ─────────────────
1. Click camera on map   →    POST /api/grid/assign → DB update
2. Video loads in slot   →    GET /api/stream/<id>  → HLS → JPEG frames
3. Toggle "AI: ON"       →    POST /api/toggle_analytics → AnalyticsEngine created
4. YOLO runs every 3rd   →    Frame → resize(640×480) → predict() → detections
5. ANPR on vehicles      →    Crop vehicle → EasyOCR → plate text
6. Watchlist match       →    plate ∈ watchlist? → log_alert() → UI flash
7. Event logged          →    INSERT INTO events → visible on /events page
8. User clears slot      →    GC loop stops analytics engine, frees CPU
```

---

## 4. Core Design Decisions

### 4.1 Grid-Slot Analytics (Resource-Efficient)

**Problem:** Running YOLO on 30+ cameras simultaneously on CPU-only hardware causes thermal throttling and system crashes.

**Solution:** Analytics runs ONLY on cameras assigned to the 2×2 video grid slots (max 4 concurrent). The analytics engine shares frames from the already-connected SentinelStreamEngine — zero duplicate connections, zero extra bandwidth.

```
Startup:  0 analytics engines running
Per-slot: 1 analytics engine per grid camera with AI: ON
Max load: 4 × YOLOv8 Nano @ 640×480 on CPU ≈ acceptable latency
Cleanup:  GC loop destroys engines when slot is cleared or camera becomes stale
```

### 4.2 Frame Sharing Architecture

```
SentinelStreamEngine         AnalyticsEngine
        │                           │
        │  self.frame_bytes         │  np.frombuffer()
        │  (JPEG bytes)     →       │  cv2.imdecode()
        │                           │  → numpy frame
        │                           │  → YOLO predict()
        │                           │  → log_event()
```

**Benefit:** One RTSP/HLS connection per camera, not two. Saves bandwidth and reduces connection failures.

### 4.3 Database Schema (SQLite — DBMS principles)

```sql
-- CCTV Registry (Model 1)
CREATE TABLE cctv_registry (
    cam_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    location TEXT,
    latitude REAL,
    longitude REAL,
    rtsp_url TEXT,
    hls_url TEXT,
    fallback_url TEXT,
    width INTEGER DEFAULT 640,
    height INTEGER DEFAULT 480,
    grid_slot INTEGER DEFAULT -1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Watchlist Records
CREATE TABLE watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL UNIQUE,
    owner_name TEXT,
    vehicle_type TEXT,
    reason TEXT,
    priority INTEGER DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vehicle Detections (ANPR results)
CREATE TABLE vehicle_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    cam_id TEXT NOT NULL,
    confidence REAL,
    is_watchlist_hit BOOLEAN DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cam_id) REFERENCES cctv_registry(cam_id)
);

-- Movement History (route reconstruction)
CREATE TABLE movement_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    cam_id TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cam_id) REFERENCES cctv_registry(cam_id)
);

-- Events Log (YOLO + ANPR detections)
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id TEXT NOT NULL,
    event_type TEXT NOT NULL,      -- 'yolo' or 'anpr'
    label TEXT NOT NULL,           -- 'person', 'car', 'GJ06DB4378', etc.
    confidence REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cam_id) REFERENCES cctv_registry(cam_id)
);

-- Watchlist Alerts (matches found during scanning)
CREATE TABLE watchlist_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    cam_id TEXT NOT NULL,
    owner_name TEXT,
    reason TEXT,
    priority INTEGER DEFAULT 1,
    confidence REAL,
    acknowledged BOOLEAN DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cam_id) REFERENCES cctv_registry(cam_id)
);
```

**DBMS Design Principles Applied:**
- **Normalization:** Separate tables for detections, movements, events, and alerts (3NF)
- **Referential Integrity:** Foreign keys enforce camera validity
- **Indexing:** Indexed on `cam_id`, `plate_number`, `timestamp` for query performance
- **Temporal Dedup:** Same plate at same camera within 60s = UPDATE not INSERT (avoids duplicates)
- **Fuzzy Search:** Falls back to `LIKE '%plate%'` when exact match fails

---

## 5. AI/Analytics Pipeline

### 5.1 YOLOv8 Nano Object Detection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | YOLOv8 Nano | Smallest YOLO variant — 3.2M params |
| Input | 640×480 | Downscaled from source — fast inference on CPU |
| Classes | person, bicycle, car, motorcycle, bus, truck | Relevant for surveillance |
| Confidence | 0.45 | Balanced precision/recall |
| Frame Skip | Every 3rd frame | Reduces CPU load by 67% |
| Device | CPU (`device='cpu'`) | No GPU required |

### 5.2 ANPR (Automatic Number Plate Recognition)

| Component | Implementation |
|-----------|---------------|
| OCR Engine | EasyOCR (lazy-loaded on first use) |
| Plate Detection | Multi-approach: adaptive threshold, Otsu, morphological ops, Canny |
| Preprocessing | Grayscale → bilateral filter → Sharpen → Resize to 300px width |
| Indian Format | Regex validation: `[A-Z]{2,3}\s?\d{1,4}\s?[A-Z]{1,3}` |
| Multi-Scale | Tries 200px, 300px, 400px, 500px widths |
| Confidence | 0.25 (vehicle crop), 0.30 (full-frame fallback) |

### 5.3 Vehicle Tracking Pipeline

```
Detection → Deduplication → GPS Lookup → Direction → Speed → Route
    │              │              │            │         │       │
    │         60s window     Haversine    Bearing    Δd/Δt   Array
    │         same camera    distance     compass   formula  of cams
    │
    └──→ vehicle_detections → movement_history → route reconstruction
```

**DSA Optimization:**
- **Haversine formula** for distance between camera GPS coordinates
- **Bearing computation** (atan2) for movement direction (N/S/E/W)
- **Speed estimation** from distance / time between consecutive detections
- **Route reconstruction** by querying movement_history ordered by timestamp

---

## 6. API Design (RESTful)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/cameras` | GET | List all cameras from registry |
| `/api/cameras/<id>` | GET | Camera details |
| `/api/grid/assign` | POST | Assign camera to grid slot |
| `/api/grid/clear/<slot>` | POST | Clear grid slot |
| `/api/stream/<id>` | GET | Progressive MP4 video stream |
| `/api/toggle_analytics` | POST | Toggle YOLO/ANPR on camera |
| `/api/analytics/status` | GET | Active analytics engines |
| `/api/events` | GET | Detection events (filtered by type/camera) |
| `/api/watchlist` | GET/POST | Watchlist management |
| `/api/watchlist/alerts` | GET | Watchlist match alerts |
| `/api/plates/suggest?q=` | GET | Plate autocomplete for search |
| `/api/vehicle/search?plate=` | GET | Vehicle search + route |

---

## 7. Frontend Architecture

| Page | Purpose | Key Features |
|------|---------|--------------|
| Dashboard (`/`) | Command centre | GIS map (Leaflet), 2×2 video wall, live detection feed |
| Events (`/events`) | Full event log | Filterable table, type/camera search, auto-refresh |
| ANPR Alerts (`/anpr-alerts`) | Plate alerts | Watchlist hits, acknowledged/unacknowledged |
| Vehicle Search (`/vehicle-search`) | Track vehicles | Plate input, route map, movement timeline |
| Watchlist (`/watchlist`) | Manage entries | Add/remove plates, owner info, priority |

**Tech Stack:** Flask (Python), TailwindCSS, Leaflet.js, HLS.js, vanilla JavaScript (no React dependency — lightweight)

---

## 8. Scalability Strategy (→ 80,000 cameras)

| Layer | Current (Prototype) | Production Scale |
|-------|-------------------|-----------------|
| **Database** | SQLite (single file) | PostgreSQL + PostGIS |
| **Streaming** | Per-camera FFmpeg | Distributed relay nodes (edge computing) |
| **Analytics** | CPU YOLOv8 Nano | GPU clusters (NVIDIA T4/A10) |
| **Search** | SQL LIKE queries | Elasticsearch for plate/event search |
| **Frontend** | Single Flask app | React SPA + WebSocket for real-time |
| **Deployment** | Single machine | Kubernetes cluster with auto-scaling |
| **Storage** | Ephemeral (no recording) | Tiered: Hot (SSD 7d) → Warm (HDD 30d) → Cold (S3 90d) |

**Edge Computing Model:**
```
Region (e.g., Ahmedabad)
├── Edge Node 1 (500 cameras) → Local YOLO → Metadata → Central DB
├── Edge Node 2 (500 cameras) → Local YOLO → Metadata → Central DB
└── Regional Hub → Aggregated alerts → State Command Centre
```

---

## 9. Security Considerations

| Aspect | Implementation |
|--------|---------------|
| Feed Access | HTTPS only for all camera connections |
| Authentication | Role-based access control (RBAC) per department |
| Data Privacy | Camera metadata only — no video storage by default |
| API Security | Token-based auth for inter-service communication |
| Audit Trail | All events/queries logged with timestamps and user IDs |
| Encryption | TLS 1.3 for data in transit, AES-256 for data at rest |

---

## 10. Cost-Benefit Analysis

| Metric | Without Sentinel | With Sentinel |
|--------|-----------------|---------------|
| Monitoring staff per command centre | 8–12 operators | 2–3 operators |
| Camera coverage visibility | ~30% (per-dept silos) | 100% (unified registry) |
| Alert response time | 15–30 min (manual) | <30 seconds (automated) |
| Vehicle tracking | Manual, days | Real-time, seconds |
| Infrastructure cost | 26 separate systems | 1 unified platform |
| Estimated annual saving | — | ₹2–3 Cr per district (staff + ops) |

---

## 11. Technology Stack Summary

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Backend | Python Flask | Rapid prototyping, ML ecosystem |
| Database | SQLite (proto) → PostgreSQL | Lightweight → Production-ready |
| AI/ML | YOLOv8 Nano + EasyOCR | Open source, CPU-friendly |
| GIS | Leaflet.js + OpenStreetMap | Free, no API key required |
| Video | FFmpeg + OpenCV | Industry standard, protocol support |
| Frontend | TailwindCSS + Vanilla JS | No build step, lightweight |
| Streaming | HLS.js + Progressive MP4 | Browser-native, fallback support |

---

## 12. Submission Deliverables

| Deliverable | Status |
|-------------|--------|
| Model 1: Centralised CCTV Registry & GIS Mapping | Complete |
| Model 2: Unified Viewing & Analytics | Complete |
| Working prototype on government feeds | Complete (30 cameras) |
| ANPR demonstration | Complete (GJ06DB4378, MH12AB5678 verified) |
| Watchlist cross-referencing + alerts | Complete |
| Vehicle tracking + route reconstruction | Complete |
| Searchable metadata dashboard | Complete |
| HLD Document | This document |
| Solution Presentation | Separate PDF |
| Demo video | To record when firewall is lifted |

---

*Prepared for Gujarat Police Innovation Hackathon 2026*
*Sentinel VMS — Team Sentinel*
