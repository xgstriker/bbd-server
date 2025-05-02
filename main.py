from flask import Flask
from routes.detect.text import text_bp
from routes.detect.object import detect_bp
from routes.detect.money import money_bp
from routes.data import data_bp
from routes.model import model_bp
import os
import sqlite3
from config import UPLOAD_FOLDER, DATABASE
app = Flask(__name__)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open("create.sql", "r") as f:
    schema = f.read()

conn = sqlite3.connect(DATABASE, check_same_thread=False)
cursor = conn.cursor()
cursor.executescript(schema)
conn.commit()
conn.close()

app.register_blueprint(text_bp)
app.register_blueprint(detect_bp)
app.register_blueprint(money_bp)
app.register_blueprint(data_bp)
app.register_blueprint(model_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
