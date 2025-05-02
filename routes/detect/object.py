from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import datetime
import sqlite3
import os
from config import DATABASE, UPLOAD_FOLDER, object_model, allowed_statuses
from functions import classify_confidence, assign_status_to_detections

detect_bp = Blueprint("detect", __name__)


def filter_detections(detections):
    return [det for det in detections if det.get("status") in allowed_statuses]


@detect_bp.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)

        results = object_model(filepath)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf)
            status = classify_confidence(conf)
            cls_index = int(box.cls)

            class_name = object_model.names.get(cls_index, f"unknown_{cls_index}")
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": list(map(int, box.xyxy[0].tolist())),
                "status": status
            })

        image_status = assign_status_to_detections(detections)
        filtered_detections = filter_detections(detections)
        # save_to_db(filepath, filtered_detections, image_status)

        # return jsonify({
        #     "detections": [d["class"] for d in filtered_detections]
        # })

        return jsonify(detections)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
