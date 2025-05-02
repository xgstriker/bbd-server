from ultralytics import YOLO

object_model = YOLO("model/object/yolo11x.pt")
# text_model = YOLO("model/text/yolo8hw.pt")
money_model = YOLO("model/money/yolo11md.pt")
UPLOAD_FOLDER = "training_data/images"
DATABASE = "database.db"
allowed_statuses = ("Good", "Middle")
