from flask import Blueprint, request
from functions import object_detection, text_detection
from config import object_model, money_model
from auth_utils import token_required

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/detect", methods=["POST"])
# @token_required
def detect():
    return object_detection(request, "Object", object_model)


@detect_bp.route("/detect-money", methods=["POST"])
# @token_required
def detect_money():
    return object_detection(request, "Money", money_model)


@detect_bp.route("/detect-text", methods=["POST"])
# @token_required
def detect_text():
    return text_detection(request)
