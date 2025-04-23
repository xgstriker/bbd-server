# 🧠 YOLO-Based Object Detection API for Partially Blind Assistance

This project is a Flask-based API that uses [YOLOv8](https://github.com/ultralytics/ultralytics) for real-time object detection. It stores detection results in a SQLite database, supports manual correction of labels, and allows retraining the model with improved data — all through clean API endpoints.

---

## 🚀 Features

- Upload and detect objects in images
- Store detections in a structured database
- Automatically assign confidence-based status (Good / Middle / Faulty)
- Review and correct detections
- Retrain the model with corrected labels
- Automatically backup and enhance the current model

---

## 🛠️ Requirements

- Python 3.8+
- `pip install -r requirements.txt`
- YOLOv8 model file (`yolov8n.pt`) in root directory
- Create DB schema from `create.sql`

---

## ▶️ Running the Server

Start the Flask app using:
```bash
python app.py
```

By default it runs on: `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
project/
├── app.py                 # Flask app runner
├── create.sql             # DB schema
├── database1.db           # SQLite database
├── yolov8n.pt             # YOLOv8 model
├── uploads/               # Incoming images
├── training_data/
│   ├── images/            # Stored training images
│   ├── labels/            # YOLO format labels
│   └── dataset.yaml       # YOLO config
├── models_backup/         # Old models before retrain
├── routes/
│   ├── detect.py          # /detect route
│   ├── data.py            # /data get/post
│   └── model.py           # /model/update
└── config.py              # Central settings
```

---

## ⚙️ API Endpoints

### 🔍 `POST /detect`
Detect objects in uploaded image and store in DB.

**Usage:**
```bash
curl -X POST http://127.0.0.1:5000/detect -F "image=@your_image.png"
```
**Returns:**
```json
{
  "detections": ["car", "person"]
}
```

---

### 📥 `GET /data`
Returns all stored image data + objects.

**Usage:**
```bash
curl http://127.0.0.1:5000/data
```

**Example Response:**
```json
[
  {
    "image_id": 1,
    "title": "20250413_181145_1",
    "datetime": "2025-04-13T18:11:45",
    "path": "training_data/images/20250413_181145_1.png",
    "status": "Good",
    "image_url": "http://127.0.0.1:5000/data/1",
    "objects": [
      {
        "id": 3,
        "class": "car",
        "confidence": 0.92,
        "bbox": [100, 200, 300, 400],
        "status": "Good"
      }
    ]
  }
]
```

---

### 📝 `POST /data`
Send corrected objects for an image.
```json
[
  {
    "image_id": 1,
    "objects": [
      {"class": "car", "confidence": 0.92, "bbox": [20, 30, 100, 200] }
    ]
  }
]
```

**Usage:**
```bash
curl -X POST http://127.0.0.1:5000/data -H "Content-Type: application/json" -d @corrections.json
```

---

### 🔁 `POST /model/update`
Retrains YOLOv8 model with current DB data. Automatically backs up previous model.

**Usage:**
```bash
curl -X POST http://127.0.0.1:5000/model/update
```

---

### 🧭 `GET /model/update/status`
Tracks training progress.
```bash
curl http://127.0.0.1:5000/model/update/status
```
**Response:**
```json
{
  "running": false,
  "message": "✅ Training complete. Model updated.",
  "result_path": "yolov8n.pt"
}
```

---

## 🧹 Cleanup Tip
If you delete files manually, be sure to remove them from the database or use a cleanup endpoint.

---

## ✅ Recommended Workflow
1. POST `/detect` to upload and detect
2. GET `/data` to review
3. POST `/data` to correct
4. POST `/model/update` to retrain
5. GET `/model/update/status` to confirm

---

## 📬 Questions?
Use the API or reach out in comments to expand features like:
- Model versioning
- Confidence tuning
- Custom dataset merging

Happy building! 🚀

