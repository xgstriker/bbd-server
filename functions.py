import sqlite3
import os
import datetime
from config import DATABASE, allowed_statuses


def classify_confidence(conf):
    if conf <= 0.5:
        return "Faulty"
    elif conf <= 0.8:
        return "Middle"
    return "Good"


def assign_status_to_detections(detections):
    statuses = [classify_confidence(d["confidence"]) for d in detections]
    if "Faulty" in statuses:
        return "Faulty"
    elif "Middle" in statuses:
        return "Middle"
    return "Good"


def filter_detections(detections):
    return [det for det in detections if det.get("status") in allowed_statuses]


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

    conn = sqlite3.connect(DATABASE, check_same_thread=False)
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
        INSERT INTO Image (Title, Extension, Type, DateTime, Path, Status, Text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        title,
        extension_id,
        type_id,
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
