import threading
import os
import shutil
import cv2
import sqlite3
import datetime
from flask import Blueprint, jsonify
from config import DATABASE, model

model_bp = Blueprint("model", __name__)
training_status = {"running": False, "message": "Idle", "result_path": None}


@model_bp.route("/model/update", methods=["POST"])
def update_model():
    if training_status["running"]:
        return jsonify({"error": "Training already in progress"}), 400

    def run_training():
        try:
            training_status["running"] = True
            training_status["message"] = "Training in progress..."

            # 1. Paths
            training_dir = "training_data"
            images_dir = os.path.join(training_dir, "images")
            labels_dir = os.path.join(training_dir, "labels")
            model_path = "yolov8n.pt"

            # 2. Backup old model
            if os.path.exists(model_path):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"models_backup/yolov8n_{timestamp}.pt"
                os.makedirs("models_backup", exist_ok=True)
                shutil.copy(model_path, backup_path)
                training_status["message"] = f"Backup created: {backup_path}"

            # 3. Clean labels
            if os.path.exists(labels_dir):
                shutil.rmtree(labels_dir)
            os.makedirs(labels_dir)

            # 4. Build new labels from DB
            conn = sqlite3.connect(DATABASE, check_same_thread=False)
            cursor = conn.cursor()
            class_map = {}
            class_id = 0

            cursor.execute("SELECT ID, Path FROM Image")
            for image_id, path in cursor.fetchall():
                filename = os.path.basename(path)
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                cursor.execute("""
                    SELECT o.Name, o.x1, o.y1, o.x2, o.y2
                    FROM Object o
                    JOIN ImageObjectLink l ON l.Object = o.ID
                    WHERE l.Image = ?
                """, (image_id,))
                label_lines = []
                for name, x1, y1, x2, y2 in cursor.fetchall():
                    if name not in class_map:
                        class_map[name] = class_id
                        class_id += 1
                    cls_id = class_map[name]
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

            conn.close()

            # 5. Write dataset.yaml
            with open(os.path.join(training_dir, "dataset.yaml"), "w") as f:
                f.write(f"path: {os.path.abspath(training_dir)}\n")
                f.write("train: images\nval: images\nnames:\n")
                for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
                    f.write(f"  {idx}: {name}\n")

            try:
                model.train(
                    data=os.path.join(training_dir, "dataset.yaml"),
                    epochs=10,
                    imgsz=1024
                )

                best_path = "runs/detect/train/weights/best.pt"
                if os.path.exists(best_path):
                    shutil.copy(best_path, model_path)
                    training_status["message"] = "✅ Training complete. Model updated."
                    training_status["result_path"] = model_path
                else:
                    training_status["message"] = "⚠️ Training ran, but best.pt not found."

            except Exception as train_error:
                training_status["message"] = f"❌ Training failed: {train_error}"


        except Exception as e:
            training_status["message"] = f"❌ Error: {e}"
        finally:
            training_status["running"] = False

    threading.Thread(target=run_training).start()
    return jsonify({"message": "Training started"})


@model_bp.route("/model/update/status", methods=["GET"])
def get_update_status():
    return jsonify(training_status)
