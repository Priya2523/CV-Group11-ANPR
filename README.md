# 🚗 Smart ANPR System — Team 11
**Automatic Number Plate Recognition for Gated Communities & Parking**

> PGDM Artificial Intelligence & Data Science | Computer Vision Project  
> Adani Institute of Digital Technology Management 

---

## 👥 Team Members
| Name | Branch |
|---|---|
| Srashti Soni | AI & DS |
| Dwiti Budh | AI & DS |
| Digesh Patel | AI & DS |
| Ravindra Tanwar | AI & DS |
| Priya A | AI & DS |

---

## 📌 Project Overview
An end-to-end intelligent vehicle entry management system that automatically detects, reads, and authenticates Indian number plates in real-time — under any lighting condition, from multiple camera angles, with a full analytics and alert dashboard.

**Powered by:** YOLOv8 + EasyOCR + Flask

---

## ✨ Features
- 🎥 **Live Camera Detection** — Real-time MJPEG stream with YOLOv8
- 🔤 **Indian Plate OCR** — EasyOCR with CLAHE + fuzzy correction
- 🌙 **Night Mode** — Auto gamma/CLAHE pipeline below brightness 85
- 📷 **Multi-Camera** — Up to 4 simultaneous streams
- ⚡ **Speed Estimation** — Zone-based overspeed alerts
- 📊 **Analytics Dashboard** — 8 chart types including anomaly detection
- 🔒 **Access Control** — Allow/Deny with blacklist support
- 🅿️ **Parking Slot Management** — Auto slot assign/free

---

## 📈 Performance
| Metric | Value |
|---|---|
| Plate Detection Accuracy | ~95% |
| OCR Read Accuracy | ~88% |
| mAP@0.5 | 0.91 |
| End-to-End Detection | < 2s |
| Live Stream FPS | 25 FPS |
| Fuzzy Match Success | ~96% |

---

## 🛠️ Tech Stack
- **YOLOv8** (Ultralytics) — plate detection
- **EasyOCR** — character recognition
- **Flask** — REST API + MJPEG streaming
- **OpenCV** — image enhancement
- **SQLite** — embedded database
- **Vanilla HTML/CSS/JS** — frontend dashboard

---



---


---



## 🚀 How to Run

### 1. Install dependencies
```bash
pip install flask opencv-python ultralytics easyocr
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

> ⚠️ **Note:** `yolov8m.pt` model weights are not included due to GitHub file size limits (50MB).  
> Download from: [Ultralytics YOLOv8](https://github.com/ultralytics/assets/releases) 

---

## 📁 Project Structure
```
CV-Group11-ANPR/
├── app.py                 # Main Flask application
├── analytics.py           # Analytics & charts
├── database.py            # SQLite DB operations
├── live_detection.py      # Camera + YOLO detection
├── manual_detection.py    # Image upload detection
├── ocr_module.py          # EasyOCR reader
├── yolo_model.py          # YOLOv8 model loader
├── night_mode.py          # Night enhancement pipeline
├── speed_estimator.py     # Speed zone estimation
├── multi_camera.py        # Multi-stream support
├── utils.py               # Slot management + fuzzy match
├── vehicle_attributes.py  # HSV color + type detection
├── best.pt                # Trained YOLO model weights
├── requirements.txt       # Python dependencies
└── templates/             # HTML frontend
```



## 📸  App Screenshots

```

---

## Step 3 — Upload files


**Batch 1 — Python files** (select all `.py` files):
`analytics.py`, `app.py`, `database.py`, `live_detection.py`, `manual_detection.py`, `multi_camera.py`, `night_mode.py`, `ocr_module.py`, `speed_estimator.py`, `utils.py`, `vehicle_attributes.py`, `yolo_model.py`

**Batch 2 — Other files:**
`best.pt`, `requirements.txt`

**Batch 3 — Folders:**
Upload `templates/` folder contents



---



---

## 📁 Project Structure
```
CV-Group11-ANPR/
├── app.py                 # Main Flask application
├── analytics.py           # Analytics & charts
├── database.py            # SQLite DB operations
├── live_detection.py      # Camera + YOLO detection
├── manual_detection.py    # Image upload detection
├── ocr_module.py          # EasyOCR reader
├── yolo_model.py          # YOLOv8 model loader
├── night_mode.py          # Night enhancement pipeline
├── speed_estimator.py     # Speed zone estimation
├── multi_camera.py        # Multi-stream support
├── utils.py               # Slot management + fuzzy match
├── vehicle_attributes.py  # HSV color + type detection
├── best.pt                # Trained YOLO model weights
├── requirements.txt       # Python dependencies
└── templates/             # HTML frontend



