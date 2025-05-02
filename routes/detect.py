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


def save_to_db(image_path, detections, image_status_title):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()

    filename = os.path.basename(image_path)
    title = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1].lstrip(".")

    # Get or insert extension
    cursor.execute("SELECT ID FROM Extension WHERE Title = ?", (extension,))
    result = cursor.fetchone()
    extension_id = result[0] if result else cursor.execute(
        "INSERT INTO Extension (Title) VALUES (?)", (extension,)
    ) or cursor.lastrowid

    # Get status ID
    cursor.execute("SELECT ID FROM Status WHERE Title = ?", (image_status_title,))
    status_row = cursor.fetchone()
    image_status_id = status_row[0] if status_row else None

    timestamp = datetime.datetime.now()
    cursor.execute("""
        INSERT INTO Image (Title, Extension, DateTime, Path, Status)
        VALUES (?, ?, ?, ?, ?)
    """, (title, extension_id, timestamp, image_path, image_status_id))
    image_id = cursor.lastrowid

    for det in detections:
        name = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        obj_status_title = det["status"]

        cursor.execute("SELECT ID FROM Status WHERE Title = ?", (obj_status_title,))
        result = cursor.fetchone()
        obj_status_id = result[0] if result else None

        cursor.execute("""
            INSERT INTO Object (Name, Detection, x1, y1, x2, y2, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, conf, x1, y1, x2, y2, obj_status_id))
        object_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO ImageObjectLink (Image, Object)
            VALUES (?, ?)
        """, (image_id, object_id))

    conn.commit()
    conn.close()


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
        save_to_db(filepath, filtered_detections, image_status)

        # return jsonify({
        #     "detections": [d["class"] for d in filtered_detections]
        # })

        return jsonify(detections)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
