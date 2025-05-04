import cv2
from flask import jsonify
import sqlite3
import os
import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from config import (
    upload_folders,
    DATABASE, allowed_statuses, detection_statuses, ocr_reader, SCHEMA_FILE
)


def _get_conn():
    return sqlite3.connect(DATABASE, check_same_thread=False)


def initiate_db():
    with open(SCHEMA_FILE, "r") as f:
        schema = f.read()

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.executescript(schema)
    conn.commit()
    conn.close()


def save_image(file, upload_type):
    # Ensure folders exists
    for folder in upload_folders.values():
        Path(folder).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{secure_filename(file.filename)}"

    # Build path using pathlib
    filepath = (Path(upload_folders[upload_type]) / filename).as_posix()
    file.save(filepath)

    return filepath


def classify_confidence(conf):
    return (
        "Faulty" if conf <= 0.5 else
        "Middle" if conf <= 0.8 else
        "Good"
    )


def assign_status_to_detections(detections):
    image_statuses = [classify_confidence(d["confidence"]) for d in detections]
    for detection_status in reversed(detection_statuses):  # check from worst to best
        if detection_status in image_statuses:
            return detection_status
    return detection_statuses[0]  # fallback to "Good"


def filter_detections(detections):
    return [det for det in detections if det.get("status") in allowed_statuses]


def get_detections(results, model):
    detections = []

    for box in results.boxes:
        conf = float(box.conf)
        status = classify_confidence(conf)
        cls_index = int(box.cls)

        class_name = model.names.get(int(box.cls), f"unknown_{cls_index}")
        detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": list(map(int, box.xyxy[0].tolist())),
            "status": status
        })

    return detections


def save_to_db(
        image_path: str,
        type_title: str,
        extension_title: str = None,
        status_title: str = None,
        detections: list = None,
        text: str = None
):
    """
    Save detection or text data to the database.

    Parameters:
    - image_path: path to the saved image
    - type_title: 'Object', 'Text', or 'Money'
    - extension_title: file extension (e.g., 'jpg')
    - status_title: image status ('Good', etc.)
    - detections: list of detection dictionaries (or None)
    - text: recognized text (or None)
    """

    conn = _get_conn()
    cursor = conn.cursor()

    # Derive filename/title/extension
    filename = os.path.basename(image_path)
    title = os.path.splitext(filename)[0]
    extension = extension_title or os.path.splitext(filename)[1].lstrip(".")

    # === Extension: lookup or insert
    cursor.execute("SELECT ID FROM Extension WHERE Title = ?", (extension,))
    ext_row = cursor.fetchone()
    if ext_row:
        extension_id = ext_row[0]
    else:
        cursor.execute("INSERT INTO Extension (Title) VALUES (?)", (extension,))
        extension_id = cursor.lastrowid

    # === Type lookup
    cursor.execute("SELECT ID FROM Type WHERE Title = ?", (type_title,))
    type_row = cursor.fetchone()
    type_id = type_row[0] if type_row else None

    # === Status lookup (for the image as a whole)
    status_id = None
    if status_title:
        cursor.execute("SELECT ID FROM Status WHERE Title = ?", (status_title,))
        s = cursor.fetchone()
        status_id = s[0] if s else None

    # === Insert into Image, converting timestamp to a string
    timestamp_str = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    cursor.execute("""
        INSERT INTO Image (Title, Extension, Type, ReadyForTraining, DateTime, Path, Status, Text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        title,
        extension_id,
        type_id,
        False,
        timestamp_str,
        image_path,
        status_id,
        text
    ))
    image_id = cursor.lastrowid

    # === Insert detections if provided
    if detections:
        for det in detections:
            name = det["class"]
            conf = float(det["confidence"])
            x1, y1, x2, y2 = map(int, det["bbox"])
            obj_status = det.get("status")

            # Lookup status ID for this object
            obj_status_id = None
            if obj_status:
                cursor.execute("SELECT ID FROM Status WHERE Title = ?", (obj_status,))
                srow = cursor.fetchone()
                obj_status_id = srow[0] if srow else None

            # Insert into Object
            cursor.execute("""
                INSERT INTO Object (Name, Detection, x1, y1, x2, y2, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, conf, x1, y1, x2, y2, obj_status_id))
            object_id = cursor.lastrowid

            # Link Image ↔ Object
            cursor.execute("""
                INSERT INTO ImageObjectLink (Image, Object)
                VALUES (?, ?)
            """, (image_id, object_id))

    conn.commit()
    conn.close()
    print(f"✅ Saved to DB (Type: {type_title}, Image ID: {image_id})")


def object_detection(request, upload_type, model):
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Save image and get filepath
        filepath = save_image(file, upload_type)

        # Money detection
        results = model(filepath)[0]

        detections = get_detections(results, model)

        # Assign overall status
        image_status = assign_status_to_detections(detections)

        # Filter out faulty detections if needed
        filtered_detections = filter_detections(detections)

        # Save to database
        save_to_db(
            image_path=filepath,
            type_title=upload_type,
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


def text_detection(request):
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Save image and get filepath
        filepath = save_image(file, "Text")

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
