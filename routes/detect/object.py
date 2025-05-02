from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import datetime
from pathlib import Path
from config import UPLOAD_FOLDER, object_model
from functions import classify_confidence, assign_status_to_detections, save_to_db, filter_detections

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400


        # Ensure folder exists
        Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

        file = request.files["image"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{secure_filename(file.filename)}"

        # Build path using pathlib
        filepath = Path(UPLOAD_FOLDER) / filename
        file.save(filepath)

        # Ensure forward slashes for DB, YAML, JSON, etc.
        filepath = filepath.as_posix()

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

        # Assign overall image status
        image_status = assign_status_to_detections(detections)

        # Filter based on allowed_statuses
        filtered_detections = filter_detections(detections)

        # Save to database
        save_to_db(
            image_path=filepath,
            type_title="Object",
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
