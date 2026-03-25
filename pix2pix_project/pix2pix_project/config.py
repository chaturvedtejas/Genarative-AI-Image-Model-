"""
config.py — Central configuration for Pix2Pix Image Translation
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ──────────────────────────────────────────────────
DATASET_PATH   = os.path.join(BASE_DIR, "dataset")
TRAIN_A        = os.path.join(DATASET_PATH, "train", "A")   # Source domain (sketch/day)
TRAIN_B        = os.path.join(DATASET_PATH, "train", "B")   # Target domain (photo/night)
VAL_A          = os.path.join(DATASET_PATH, "val",   "A")
VAL_B          = os.path.join(DATASET_PATH, "val",   "B")

# ── Supported translation modes ──────────────────────────────
MODES = {
    "sketch2photo": {
        "description": "Hand sketch → Realistic photo",
        "dataset_url":  "https://www.kaggle.com/datasets/shubham0204/facades-dataset-pix2pix",
        "source_label": "Sketch / Edge map",
        "target_label": "Realistic photo",
    },
    "day2night": {
        "description": "Daytime scene → Night scene",
        "dataset_url":  "https://www.kaggle.com/datasets/hayiyo/pix2pix-day-to-night",
        "source_label": "Daytime image",
        "target_label": "Night image",
    },
    "edges2shoes": {
        "description": "Edge map → Photorealistic shoe",
        "dataset_url":  "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz",
        "source_label": "Edge map",
        "target_label": "Photorealistic shoe",
    },
    "facades": {
        "description": "Building label map → Facade photo",
        "dataset_url":  "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz",
        "source_label": "Label / segmentation map",
        "target_label": "Building facade",
    },
}

# ── Image settings ────────────────────────────────────────────
IMG_HEIGHT     = 256
IMG_WIDTH      = 256
IMG_CHANNELS   = 3
BUFFER_SIZE    = 400

# ── Training hyperparameters ──────────────────────────────────
BATCH_SIZE     = 1          # Pix2Pix works best with batch=1
EPOCHS         = 150
LAMBDA_L1      = 100        # L1 loss weight (reconstruction)
GEN_LR         = 2e-4
DISC_LR        = 2e-4
BETA_1         = 0.5        # Adam momentum (standard for GANs)
BETA_2         = 0.999

# ── Checkpointing ─────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
CHECKPOINT_FREQ = 10        # Save every N epochs
SAMPLE_FREQ     = 5         # Generate sample images every N epochs

# ── Paths ─────────────────────────────────────────────────────
GENERATOR_PATH  = os.path.join(BASE_DIR, "models", "generator.h5")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
SAMPLE_DIR      = os.path.join(BASE_DIR, "logs", "samples")

# ── Flask ─────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
SECRET_KEY      = "pix2pix-secret-2024"
UPLOAD_FOLDER   = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER  = os.path.join(BASE_DIR, "static", "results")
MAX_CONTENT_MB  = 10
