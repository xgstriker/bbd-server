from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import easyocr
import os
import datetime
import cv2
from config import DATABASE
from functions import save_detection_to_db

text_bp = Blueprint("text_detect", __name__)
UPLOAD_FOLDER = "uploads/text"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

languages = [
    'en',  # English
    'de',  # German
    'fr',  # French
    'es',  # Spanish
    'it',  # Italian
    'pl',  # Polish
    'lt',  # Lithuanian
    'lv',  # Latvian
    'et',  # Estonian
    'ro',  # Romanian
    'tr',  # Turkish
    'cs',  # Czech
    'sk',  # Slovak
    'sl',  # Slovenian
    'hu'  # Hungarian
]

# EasyOCR reader setup with multiple languages
ocr_reader = easyocr.Reader(languages, gpu=False)

@text_bp.route("/detect-text", methods=["POST"])
def detect_text():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        detections = []
        full_texts = []

        # OCR with bounding boxes
        ocr_results = ocr_reader.readtext(img)
        for bbox, text, _ in ocr_results:
            x1, y1 = map(int, bbox[0])
            x2, y2 = map(int, bbox[2])
            cleaned = text.strip()
            if cleaned:
                detections.append({
                    "class": "text",
                    "confidence": 1.0,
                    "bbox": [x1, y1, x2, y2]
                })
                full_texts.append(cleaned)

        # Save into DB
        # save_detection_to_db(
        #     image_path=filepath,
        #     detections=detections,
        #     type_title="Text",
        #     full_text="\n".join(full_texts)
        # )

        return jsonify({
            "text": "\n".join(full_texts),
            "regions": full_texts
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
