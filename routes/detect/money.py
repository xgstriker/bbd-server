from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from config import money_model
from functions import (
    save_to_db,
    classify_confidence,
    assign_status_to_detections,
    filter_detections
)
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
            conf = float(box.conf)
            label = money_model.names.get(int(box.cls), f"unknown_{int(box.cls)}")
            status = classify_confidence(conf)

            detections.append({
                "class": label,
                "confidence": conf,
                "bbox": list(map(int, box.xyxy[0].tolist())),
                "status": status
            })

        # Assign overall status
        image_status = assign_status_to_detections(detections)

        # Filter out faulty detections if needed
        filtered_detections = filter_detections(detections)

        # Save to database
        save_to_db(
            image_path=filepath,
            type_title="Money",
            status_title=image_status,
            detections=filtered_detections
        )

        return jsonify({
            "status": image_status,
            "detections": [d["class"] for d in filtered_detections]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
