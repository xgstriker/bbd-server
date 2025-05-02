from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import easyocr
import os
import datetime
import cv2
from functions import save_to_db

text_bp = Blueprint("text_detect", __name__)
UPLOAD_FOLDER = "uploads/text"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported European and nearby OCR languages
languages = [
    'en', 'de', 'fr', 'es', 'it', 'pl', 'lt', 'lv', 'et',
    'ro', 'tr', 'cs', 'sk', 'sl', 'hu'
]

ocr_reader = easyocr.Reader(languages, gpu=False)

@text_bp.route("/detect-text", methods=["POST"])
def detect_text():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Save uploaded image
        file = request.files["image"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read image and apply OCR
        img = cv2.imread(filepath)
        ocr_results = ocr_reader.readtext(img)

        regions = []
        full_texts = []

        for bbox, text, _ in ocr_results:
            cleaned = text.strip()
            if cleaned:
                full_texts.append(cleaned)
                regions.append({
                    "text": cleaned,
                    "bbox": [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])]
                })

        joined_text = "\n".join(full_texts)

        # Save to database (type='Text', no detections, only OCR text)
        save_to_db(
            image_path=filepath,
            type_title="Text",
            text=joined_text
        )

        return jsonify({
            "text": joined_text
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
