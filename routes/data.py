from flask import Blueprint, request, jsonify, url_for, send_file
import sqlite3
import os
from config import DATABASE
from functions import classify_confidence, assign_status_to_detections

data_bp = Blueprint("data", __name__)


def _get_conn():
    return sqlite3.connect(DATABASE, check_same_thread=False)


@data_bp.route("/data", methods=["GET"])
def get_data():
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            i.ID,
            i.Title,
            i.DateTime,
            s.Title   AS Status,
            e.Title   AS Extension,
            t.Title   AS Type,
            i.Text    AS Text
        FROM Image i
        LEFT JOIN Status    s ON i.Status    = s.ID
        LEFT JOIN Extension e ON i.Extension = e.ID
        LEFT JOIN Type      t ON i.Type      = t.ID
        ORDER BY i.DateTime DESC
    """)
    rows = cursor.fetchall()

    out = []
    for img_id, title, dt, status, ext, typ, text in rows:
        # build a download URL for this image
        image_url = url_for("data.download_image", image_id=img_id, _external=True)

        # pull its objects
        cursor.execute("""
            SELECT
                o.ID,
                o.Name,
                o.Detection,
                o.x1, o.y1, o.x2, o.y2,
                s.Title AS ObjStatus
            FROM Object o
            JOIN ImageObjectLink l ON l.Object = o.ID
            LEFT JOIN Status s ON o.Status = s.ID
            WHERE l.Image = ?
        """, (img_id,))
        objs = [
            {
                "id": oid,
                "class": name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "status": obj_st
            }
            for (oid, name, conf, x1, y1, x2, y2, obj_st)
            in cursor.fetchall()
        ]

        out.append({
            "image_id": img_id,
            "title": title,
            "datetime": dt,
            "status": status,
            "extension": ext,
            "type": typ,
            "text": text,
            "image_url": image_url,
            "objects": objs
        })

    conn.close()
    return jsonify(out)


@data_bp.route("/data/image/<int:image_id>", methods=["GET"])
def download_image(image_id):
    """
    Lookup the image path by ID and send it back as a file.
    """
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT Path FROM Image WHERE ID = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Image not found"}), 404

    path = row[0]
    if not os.path.isfile(path):
        return jsonify({"error": "File missing on disk"}), 404

    # send_file will guess the mimetype; as_attachment=False streams inline
    return send_file(path, as_attachment=False)


@data_bp.route("/data", methods=["POST"])
def update_data():
    updates = request.get_json()
    if not isinstance(updates, list):
        return jsonify({"error": "Expected a list of image updates"}), 400

    conn = _get_conn()
    cursor = conn.cursor()

    for img in updates:
        image_id = img.get("image_id")
        new_objects = img.get("objects", [])
        new_text = img.get("text", None)
        new_type = img.get("type", None)
        new_ext = img.get("extension", None)

        if image_id is None:
            continue

        # 1) delete old objects & links
        cursor.execute(
            "SELECT Object FROM ImageObjectLink WHERE Image = ?",
            (image_id,)
        )
        old_ids = [r[0] for r in cursor.fetchall()]
        if old_ids:
            ph = ",".join("?" * len(old_ids))
            cursor.execute(f"DELETE FROM Object WHERE ID IN ({ph})", old_ids)
        cursor.execute("DELETE FROM ImageObjectLink WHERE Image = ?", (image_id,))

        # 2) insert new objects & update links
        for det in new_objects:
            nm = det["class"]
            conf = float(det["confidence"])
            x1, y1, x2, y2 = map(int, det["bbox"])
            stitle = classify_confidence(conf)

            cursor.execute(
                "SELECT ID FROM Status WHERE Title = ?",
                (stitle,)
            )
            srow = cursor.fetchone()
            sid = srow[0] if srow else None

            cursor.execute("""
                INSERT INTO Object (Name, Detection, x1, y1, x2, y2, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (nm, conf, x1, y1, x2, y2, sid))
            obj_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO ImageObjectLink (Image, Object) VALUES (?, ?)",
                (image_id, obj_id)
            )

        # 3) recompute & update image-level status
        img_st = assign_status_to_detections(new_objects)
        cursor.execute("SELECT ID FROM Status WHERE Title = ?", (img_st,))
        row = cursor.fetchone()
        img_sid = row[0] if row else None

        # 4) update Image row (Status, Text, Type, Extension) if provided
        fields, params = [], []
        if img_sid is not None:
            fields.append("Status = ?");
            params.append(img_sid)
        if new_text is not None:
            fields.append("Text = ?");
            params.append(new_text)
        if new_type is not None:
            cursor.execute("SELECT ID FROM Type WHERE Title = ?", (new_type,))
            r = cursor.fetchone()
            tid = r[0] if r else None
            fields.append("Type = ?");
            params.append(tid)
        if new_ext is not None:
            cursor.execute("SELECT ID FROM Extension WHERE Title = ?", (new_ext,))
            r = cursor.fetchone()
            if r:
                eid = r[0]
            else:
                cursor.execute(
                    "INSERT INTO Extension (Title) VALUES (?)",
                    (new_ext,)
                )
                eid = cursor.lastrowid
            fields.append("Extension = ?");
            params.append(eid)

        if fields:
            params.append(image_id)
            sql = f"UPDATE Image SET {', '.join(fields)} WHERE ID = ?"
            cursor.execute(sql, params)

    conn.commit()
    conn.close()
    return jsonify({"success": True})



