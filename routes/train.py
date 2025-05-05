import os
import shutil
import datetime
import glob
from pathlib import Path
from threading import Thread

from flask import Blueprint, jsonify
from ultralytics import YOLO
from PIL import Image
import yaml
from auth_utils import token_required, admin_required

from auth_utils import token_required
from config import (
    BACKUP_DIR,
    TRAINING_DATA_DIR,
    RUNS_DIR,
    MODEL_CONFIG
)
from functions import _get_conn

training_bp = Blueprint("training", __name__)


# ─── utility functions ───────────────────────────────────────────────────────

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _move_ready_images(cursor, model_type):
    """
    1) Move all ReadyForTraining images of this type into
       training_data/[type]/images as <ID><ext>, return list of (ID, dest_path).
    """
    # ── (A) Lookup the numeric Type.ID ────────────────────────────────
    cursor.execute("SELECT ID FROM Type WHERE Title = ?", (model_type,))
    row = cursor.fetchone()
    if not row:
        print(f"⚠️ No Type entry for '{model_type}'")
        return []
    type_id = row[0]

    # ── (B) Fetch all ReadyForTraining images of that type ────────────
    cursor.execute(
        "SELECT ID, Path FROM Image WHERE ReadyForTraining = 1 AND Type = ?",
        (type_id,),
    )
    rows = cursor.fetchall()

    dest_dir = Path(TRAINING_DATA_DIR) / model_type / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for img_id, db_path in rows:
        # ── (C) Normalize the DB path for Windows backslashes ────────
        normalized = os.path.normpath(db_path)
        src_path = Path(normalized)
        # If it’s relative, resolve against project root
        if not src_path.is_absolute():
            src_path = (Path(__file__).parent.parent / src_path).resolve()

        if not src_path.exists():
            print(f"⚠️ Skipping image {img_id}: file not found at {src_path}")
            continue

        # ── (D) Move by copy+unlink and rename to <ID><ext> ───────────
        ext = src_path.suffix or ".jpg"
        dest_path = dest_dir / f"{img_id}{ext}"
        try:
            shutil.copy2(src_path, dest_path)
            src_path.unlink()
            moved.append((img_id, dest_path))
        except Exception as e:
            print(f"❌ Failed to move image {img_id}: {e}")

    return moved




def _create_labels_for_images(cursor, model_type, image_info):
    """2) For each (ID, image_path), write a YOLO .txt label next to it."""
    labels_dir = Path(TRAINING_DATA_DIR) / model_type / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # load names→index from the model
    model = YOLO(MODEL_CONFIG[model_type]["path"])
    name2idx = {v: k for k, v in model.names.items()}

    for img_id, image_path in image_info:
        # fetch dets
        cursor.execute(
            """
            SELECT o.Name, o.x1, o.y1, o.x2, o.y2
              FROM Object o
              JOIN ImageObjectLink l ON l.Object = o.ID
             WHERE l.Image = ?
            """,
            (img_id,),
        )
        dets = cursor.fetchall()

        # get image size
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size

        lines = []
        for cls_name, x1, y1, x2, y2 in dets:
            idx = name2idx.get(cls_name, 0)
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # write label file alongside the image
        label_file = labels_dir / f"{img_id}.txt"
        label_file.write_text("\n".join(lines))


def _make_dataset_yaml(model_type):
    """3) Emit a dataset YAML under training_data/[type]/dataset.yaml."""
    base = Path(TRAINING_DATA_DIR) / model_type
    images = (base / "images").resolve().as_posix()
    labels = (base / "labels").resolve().as_posix()
    # count classes by scanning labels
    class_ids = set()
    for f in glob.glob(f"{labels}/*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if parts:
                class_ids.add(int(parts[0]))
    nc = max(class_ids) + 1 if class_ids else 0
    names = {i: str(i) for i in range(nc)}

    cfg = {
        "train": images,
        "val": images,
        "nc": nc,
        "names": names,
    }
    out = base / "dataset_auto.yaml"
    _ensure_dir(out.parent)
    with open(out, "w") as fp:
        yaml.dump(cfg, fp)
    return out.as_posix()


def _backup_existing_model(model_type):
    """4) Copy current weights into models_backup/[type]."""
    cfg = MODEL_CONFIG[model_type]
    src = cfg["path"]
    dst_dir = Path(BACKUP_DIR) / model_type
    _ensure_dir(dst_dir)
    dst = dst_dir / f"{Path(src).stem}_{_timestamp()}.pt"
    shutil.copy2(src, dst)
    return dst


def _evaluate_and_promote(model_type, run_name):
    """
    6) Compare new vs old. If worse, move this run into was_not_worth_it/[type].
       If better, overwrite the old weights in-place.
    """
    cfg     = MODEL_CONFIG[model_type]
    run_dir = Path(RUNS_DIR) / cfg["runs"] / run_name
    new_w   = run_dir / "weights" / "best.pt"
    old_w   = Path(cfg["path"])

    # regenerate the dataset yaml (must match the training/val split)
    data_yaml = _make_dataset_yaml(model_type)

    # evaluate the old and new models
    old_metrics = YOLO(old_w).val(data=data_yaml)
    new_metrics = YOLO(new_w).val(data=data_yaml)

    # directly access box.map50
    m_old = old_metrics.box.map50
    m_new = new_metrics.box.map50

    if m_new <= m_old:
        # not worth it: archive the entire run folder
        dest = Path("was_not_worth_it") / model_type
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dest / run_name))
        print(f"📉 New model ({m_new:.4f}) ≤ old ({m_old:.4f}), archived.")
    else:
        # promote new weights
        shutil.copy2(new_w, old_w)
        print(f"📈 New model ({m_new:.4f}) > old ({m_old:.4f}), promoted.")


def _delete_trained_images(cursor, image_ids):
    """8) Delete these images + their Object rows and links from the DB."""
    if not image_ids:
        return

    ph = ",".join("?" * len(image_ids))

    # 8a) Find all Object IDs linked to those images
    cursor.execute(
        f"SELECT Object FROM ImageObjectLink WHERE Image IN ({ph})",
        image_ids
    )
    obj_ids = [r[0] for r in cursor.fetchall()]

    # 8b) Delete the objects themselves
    if obj_ids:
        oph = ",".join("?" * len(obj_ids))
        cursor.execute(
            f"DELETE FROM Object WHERE ID IN ({oph})",
            obj_ids
        )

    # 8c) Delete the links
    cursor.execute(
        f"DELETE FROM ImageObjectLink WHERE Image IN ({ph})",
        image_ids
    )

    # 8d) Finally delete the Image rows
    cursor.execute(
        f"DELETE FROM Image WHERE ID IN ({ph})",
        image_ids
    )


def _cleanup_training_dir(model_type):
    """9) Wipe training_data/[type] entirely."""
    base = Path(TRAINING_DATA_DIR) / model_type
    if base.exists():
        shutil.rmtree(base)


# ─── the master training function ────────────────────────────────────────────

def _run_training(model_type: str):
    cfg = MODEL_CONFIG[model_type]
    conn = _get_conn()
    cursor = conn.cursor()

    try:
        # 1) move & gather → now returns [(id, dest_path), …]
        image_info = _move_ready_images(cursor, model_type)

        image_ids = [img_id for img_id, _ in image_info]
        # 2) label creation takes (id, path) pairs
        _create_labels_for_images(cursor, model_type, image_info)

        data_yaml = _make_dataset_yaml(model_type)

        # 4) backup existing weights
        backup = _backup_existing_model(model_type)

        # 5) actual training
        run_name = f"{cfg['runs']}_{_timestamp()}"
        training_ok = False
        try:
            YOLO(cfg["path"]).train(
                data=data_yaml,
                epochs=50,
                project=os.path.join(RUNS_DIR, cfg["runs"]),
                name=run_name,
            )
            training_ok = True
            print(f"✅ {model_type} training completed: {run_name}")
        except Exception as train_err:
            print(f"❌ {model_type} training failed: {train_err}")

        # only steps 6–9 if training actually ran
        if training_ok:
            _evaluate_and_promote(model_type, run_name)
            cfg["reload"]()
            _delete_trained_images(cursor, image_ids)
            conn.commit()
            _cleanup_training_dir(model_type)
            print(f"✅ Post-training steps for {model_type} complete")
        else:
            print(f"⚠️ Skipping post-training steps for {model_type} because training failed")

    finally:
        conn.close()


# ─── Flask routes ───────────────────────────────────────────────────────────

def _start_training(kind):
    if kind not in MODEL_CONFIG:
        return jsonify({"error": "unknown model type"}), 400
    Thread(target=lambda: _run_training(kind), daemon=True).start()
    return jsonify({"message": f"{kind} training started", "running": True})


@training_bp.route("/train-money", methods=["POST"])
# @admin_required
def train_money():
    return _start_training("Money")


@training_bp.route("/train-object", methods=["POST"])
# @admin_required
def train_object():
    return _start_training("Object")
