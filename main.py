from flask import Flask
from routes.detect import detect_bp
from routes.data import data_bp
from routes.train import training_bp
from functions import initiate_db

app = Flask(__name__)

initiate_db()

app.register_blueprint(detect_bp)
app.register_blueprint(data_bp)
app.register_blueprint(training_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
