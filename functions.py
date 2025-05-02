import sqlite3
import os
import datetime
from config import DATABASE


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


def save_detection_to_db(
        image_path: str,
        detections: list,
        type_title: str,
        full_text: str = None
):
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

    # Get Type ID
    cursor.execute("SELECT ID FROM Type WHERE Title = ?", (type_title,))
    type_row = cursor.fetchone()
    type_id = type_row[0] if type_row else None

    # Insert image
    now = datetime.datetime.now()
    cursor.execute("""
        INSERT INTO Image (Title, Extension, Type, DateTime, Path, Text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (title, extension_id, type_id, now, image_path, full_text))
    image_id = cursor.lastrowid

    # Insert each detection
    for det in detections:
        name = det.get("class", "unknown")
        conf = det.get("confidence", 1.0)
        x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, 0, 0]))

        # Optional status
        status_title = det.get("status")
        status_id = None
        if status_title:
            cursor.execute("SELECT ID FROM Status WHERE Title = ?", (status_title,))
            result = cursor.fetchone()
            status_id = result[0] if result else None

        cursor.execute("""
            INSERT INTO Object (Name, Detection, x1, y1, x2, y2, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, conf, x1, y1, x2, y2, status_id))
        object_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO ImageObjectLink (Image, Object)
            VALUES (?, ?)
        """, (image_id, object_id))

    conn.commit()
    conn.close()
