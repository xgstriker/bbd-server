from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from config import money_model
from functions import save_detection_to_db
import os
import datetime

money_bp = Blueprint("money_detect", __name__)
UPLOAD_FOLDER = "uploads/money"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@money_bp.route("/detect-money", methods=["POST"])
def detect_money():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = money_model(filepath)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls)
            label = money_model.names.get(cls, f"unknown_{cls}")  # e.g., '10_EUR'

            detections.append({
                "class": label,
                "confidence": float(box.conf),
                "bbox": list(map(int, box.xyxy[0].tolist()))
            })

        # save_detection_to_db(
        #     image_path=filepath,
        #     detections=detections,
        #     type_title="Money"
        # )

        return jsonify({
            "detections": detections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
