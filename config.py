from ultralytics import YOLO

model = YOLO("yolov8n.pt")
UPLOAD_FOLDER = "training_data/images"
DATABASE = "database.db"
allowed_statuses = ("Good", "Middle")
