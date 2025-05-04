from flask import Blueprint, request
from functions import object_detection, text_detection
from config import object_model, money_model

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/detect", methods=["POST"])
def detect():
    return object_detection(request, "Object", object_model)


@detect_bp.route("/detect-money", methods=["POST"])
def detect_money():
    return object_detection(request, "Money", money_model)


@detect_bp.route("/detect-text", methods=["POST"])
def detect_text():
    return text_detection(request)
