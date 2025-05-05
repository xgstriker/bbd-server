from flask import Blueprint, request, jsonify
import sqlite3
from config import DATABASE
from auth_utils import (
    hash_password, verify_password, generate_token,
    token_required, admin_required
)

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username","").strip()
    password = data.get("password","").strip()
    if not username or not password:
        return jsonify({"error":"Username and password required"}),400

    hashed = hash_password(password)
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO User (Username, Password) VALUES (?,?)",
            (username, hashed)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error":"Username already exists"}),409
    conn.close()
    return jsonify({"message":"Registered as user"}),201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username","").strip()
    password = data.get("password","").strip()
    if not username or not password:
        return jsonify({"error":"Username and password required"}),400

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        "SELECT ID, Password, Role FROM User WHERE Username=?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(password, row[1]):
        return jsonify({"error":"Invalid credentials"}),401

    user_id, _, role = row
    token = generate_token(user_id, username, role)
    return jsonify({"token":token, "role":role}),200

@auth_bp.route("/register-admin", methods=["POST"])
@admin_required
def register_admin():
    """Admin-only: create another admin"""
    data = request.get_json() or {}
    username = data.get("username","").strip()
    password = data.get("password","").strip()
    if not username or not password:
        return jsonify({"error":"Username and password required"}),400

    hashed = hash_password(password)
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO User (Username, Password, Role) VALUES (?,?, 'admin')",
            (username, hashed)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error":"Username already exists"}),409
    conn.close()
    return jsonify({"message":"Registered as admin"}),201
