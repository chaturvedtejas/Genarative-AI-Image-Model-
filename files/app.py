"""
app.py — Flask Web Server for Pix2Pix Image Translation
────────────────────────────────────────────────────────────────
Routes:
  GET  /                    → Main web app
  POST /api/translate       → Upload image → get translated image
  POST /api/edge_extract    → Upload photo → get edge map
  GET  /api/modes           → List supported translation modes
  GET  /api/history         → Recent translation history
  GET  /static/...          → Serve uploaded / result images
"""

import os, uuid, json, time
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

from config import *
from inference import translate_image, extract_edges, get_generator

app = Flask(__name__)
app.config["SECRET_KEY"]       = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
history_log = []   # In-memory history (replace with DB for production)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(ext="jpg"):
    return f"{uuid.uuid4().hex}.{ext}"


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", modes=MODES)


@app.route("/api/modes")
def api_modes():
    return jsonify(MODES)


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Accepts:
      file       : image file
      mode       : translation mode key (optional, default sketch2photo)
      auto_edge  : "true" → extract edges from photo before translating
    Returns:
      { input_url, output_url, elapsed_ms, mode }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    mode      = request.form.get("mode", "sketch2photo")
    auto_edge = request.form.get("auto_edge", "false").lower() == "true"

    # Save upload
    ext       = file.filename.rsplit(".", 1)[1].lower()
    in_fname  = unique_filename(ext)
    out_fname = unique_filename("jpg")
    in_path   = os.path.join(UPLOAD_FOLDER,  in_fname)
    out_path  = os.path.join(RESULTS_FOLDER, out_fname)
    file.save(in_path)

    # Translate
    t0 = time.time()
    try:
        translate_image(in_path, out_path, auto_edge=auto_edge)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    elapsed = int((time.time() - t0) * 1000)

    # Log
    entry = {
        "id":         uuid.uuid4().hex[:8],
        "timestamp":  time.strftime("%H:%M:%S"),
        "mode":       MODES.get(mode, {}).get("description", mode),
        "input_url":  f"/static/uploads/{in_fname}",
        "output_url": f"/static/results/{out_fname}",
        "elapsed_ms": elapsed,
    }
    history_log.insert(0, entry)
    if len(history_log) > 20:
        history_log.pop()

    return jsonify(entry)


@app.route("/api/edge_extract", methods=["POST"])
def api_edge_extract():
    """Convert a photo to an edge map (for preview before translating)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    in_fname = unique_filename(ext)
    out_fname = unique_filename("jpg")
    in_path  = os.path.join(UPLOAD_FOLDER,  in_fname)
    out_path = os.path.join(RESULTS_FOLDER, out_fname)
    file.save(in_path)

    edges = extract_edges(in_path, out_path)
    return jsonify({
        "edge_url": f"/static/results/{out_fname}",
        "input_url": f"/static/uploads/{in_fname}",
    })


@app.route("/api/history")
def api_history():
    return jsonify(history_log)


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


# ─────────────────────────────────────────────────────────────
#  START
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load model at startup to avoid first-request delay
    try:
        get_generator()
    except FileNotFoundError:
        print("[!] No trained model found — train first with: python train.py")

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
