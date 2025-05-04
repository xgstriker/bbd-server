from ultralytics import YOLO
import easyocr

DATABASE = "database.db"
SCHEMA_FILE = "create.sql"

allowed_statuses = ("Good", "Middle")
detection_statuses = ("Good", "Middle", "Faulty")

# Supported European and nearby OCR languages
languages = [
    'en', 'de', 'fr', 'es', 'it', 'pl', 'lt', 'lv', 'et',
    'ro', 'tr', 'cs', 'sk', 'sl', 'hu'
]

ocr_reader = easyocr.Reader(languages, gpu=False)

upload_folders = {
    "Object": "uploads/object_images",
    "Money": "uploads/money_images",
    "Text": "uploads/text_images"
}

model_paths = {
    "Object": "model/object/yolo11x.pt",
    "Money": "model/money/yolo11md.pt"
}

BACKUP_DIR = "models_backup"
TRAINING_DATA_DIR = "training_data"
RUNS_DIR = "runs"

object_model = YOLO(model_paths["Object"])
money_model = YOLO(model_paths["Money"])


def update_object_model():
    global object_model
    object_model = YOLO(model_paths["Object"])


def update_money_model():
    global money_model
    money_model = YOLO(model_paths["Money"])


MODEL_CONFIG = {
    "Money": {
        "path": model_paths["Object"],
        "upload_folder": upload_folders["Money"],
        "runs": "train_money",
        "reload": update_money_model
    },
    "Object": {
        "path": model_paths["Money"],
        "upload_folder": upload_folders["Object"],
        "runs": "train_object",
        "reload": update_object_model
    }
}