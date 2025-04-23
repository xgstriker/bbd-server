from flask import Blueprint, request, jsonify
import sqlite3
import base64
from config import DATABASE
from functions import classify_confidence, assign_status_to_detections

data_bp = Blueprint("data", __name__)


@data_bp.route("/data", methods=["GET"])
def get_data():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT i.ID, i.Title, i.DateTime, i.Path, s.Title
        FROM Image i
        LEFT JOIN Status s ON i.Status = s.ID
    """)
    images = cursor.fetchall()

    data = []
    for img in images:
        img_id, title, dt, path, status = img

        # 🖼️ Try to read image as base64
        try:
            with open(path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Could not read image at {path}: {e}")
            image_base64 = None

        # 📦 Get linked objects
        cursor.execute("""
            SELECT o.ID, o.Name, o.Detection, o.x1, o.y1, o.x2, o.y2, s.Title
            FROM Object o
            JOIN ImageObjectLink l ON l.Object = o.ID
            LEFT JOIN Status s ON o.Status = s.ID
            WHERE l.Image = ?
        """, (img_id,))
        objects = [
            {
                "id": row[0],
                "class": row[1],
                "confidence": row[2],
                "bbox": [row[3], row[4], row[5], row[6]],
                "status": row[7]
            }
            for row in cursor.fetchall()
        ]

        data.append({
            "image_id": img_id,
            "title": title,
            "datetime": dt,
            "status": status,
            "image": image_base64,
            "objects": objects
        })

    conn.close()
    return jsonify(data)


@data_bp.route("/data", methods=["POST"])
def update_data():
    updates = request.get_json()
    if not isinstance(updates, list):
        return jsonify({"error": "Expected a list of image data"}), 400

    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()

    for image in updates:
        image_id = image.get("image_id")
        new_objects = image.get("objects", [])

        # 1. Delete old objects + links for this image
        cursor.execute("SELECT Object FROM ImageObjectLink WHERE Image = ?", (image_id,))
        object_ids = [row[0] for row in cursor.fetchall()]

        if object_ids:
            cursor.execute(f"DELETE FROM Object WHERE ID IN ({','.join('?' * len(object_ids))})", object_ids)
        cursor.execute("DELETE FROM ImageObjectLink WHERE Image = ?", (image_id,))

        # 2. Reinsert new objects + links
        for obj in new_objects:
            name = obj["class"]
            conf = obj["confidence"]
            x1, y1, x2, y2 = obj["bbox"]
            status_title = classify_confidence(conf)

            # Get status ID
            cursor.execute("SELECT ID FROM Status WHERE Title = ?", (status_title,))
            result = cursor.fetchone()
            status_id = result[0] if result else None

            cursor.execute("""
                INSERT INTO Object (Name, Detection, x1, y1, x2, y2, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, conf, x1, y1, x2, y2, status_id))
            obj_id = cursor.lastrowid

            cursor.execute("INSERT INTO ImageObjectLink (Image, Object) VALUES (?, ?)", (image_id, obj_id))

        # 3. Reassign image status
        image_status = assign_status_to_detections(new_objects)
        cursor.execute("SELECT ID FROM Status WHERE Title = ?", (image_status,))
        row = cursor.fetchone()
        status_id = row[0] if row else None

        cursor.execute("UPDATE Image SET Status = ? WHERE ID = ?", (status_id, image_id))

    conn.commit()
    conn.close()

    return jsonify({"success": True})
