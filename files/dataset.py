"""
dataset.py — tf.data pipeline for Pix2Pix
────────────────────────────────────────────────────────────────
Handles two dataset layouts:

Layout A (paired side-by-side):
  Single image file where left half = source, right half = target
  e.g. original Pix2Pix facades dataset

Layout B (separate folders):
  dataset/train/A/*.jpg  ← source images
  dataset/train/B/*.jpg  ← target images (same filenames)

Preprocessing:
  • Resize to 286×286
  • Random crop to 256×256  (train only)
  • Random horizontal flip  (train only)
  • Normalize to [-1, 1]
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from config import IMG_HEIGHT, IMG_WIDTH, BUFFER_SIZE, BATCH_SIZE


# ─────────────────────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load a single image and convert to float32 [-1, 1]."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    return img


def split_paired_image(image):
    """
    For side-by-side paired datasets:
    Left  → input (A)
    Right → target (B)
    """
    w = tf.shape(image)[1]
    w = w // 2
    input_img  = image[:, :w, :]
    target_img = image[:, w:, :]
    return input_img, target_img


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────────────────────

def random_jitter(input_img, target_img):
    """
    Augmentation as in original Pix2Pix paper:
    1. Resize to 286×286
    2. Random crop to 256×256
    3. Random horizontal flip
    """
    # Resize
    input_img  = tf.image.resize(input_img,  [286, 286], method="nearest")
    target_img = tf.image.resize(target_img, [286, 286], method="nearest")

    # Stack for synchronized crop
    stacked = tf.stack([input_img, target_img], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    input_img, target_img = cropped[0], cropped[1]

    # Random flip
    if tf.random.uniform(()) > 0.5:
        input_img  = tf.image.flip_left_right(input_img)
        target_img = tf.image.flip_left_right(target_img)

    return input_img, target_img


def normalize(input_img, target_img):
    """Scale pixel values from [0, 255] to [-1, 1]."""
    input_img  = (input_img  / 127.5) - 1.0
    target_img = (target_img / 127.5) - 1.0
    return input_img, target_img


def resize_only(input_img, target_img):
    """For val/test — resize but no random crop/flip."""
    input_img  = tf.image.resize(input_img,  [IMG_HEIGHT, IMG_WIDTH])
    target_img = tf.image.resize(target_img, [IMG_HEIGHT, IMG_WIDTH])
    return input_img, target_img


# ─────────────────────────────────────────────────────────────
#  PAIRED LAYOUT (side-by-side)
# ─────────────────────────────────────────────────────────────

def load_paired_train(image_file):
    img = load_image(image_file)
    a, b = split_paired_image(img)
    a, b = random_jitter(a, b)
    a, b = normalize(a, b)
    return a, b


def load_paired_val(image_file):
    img = load_image(image_file)
    a, b = split_paired_image(img)
    a, b = resize_only(a, b)
    a, b = normalize(a, b)
    return a, b


def make_paired_dataset(folder: str, is_train: bool = True, batch_size: int = BATCH_SIZE):
    """Build tf.data dataset from a folder of side-by-side paired images."""
    pattern = str(Path(folder) / "*.jpg")
    ds = tf.data.Dataset.list_files(pattern, shuffle=is_train)

    load_fn = load_paired_train if is_train else load_paired_val
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        ds = ds.shuffle(BUFFER_SIZE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────
#  SEPARATE FOLDER LAYOUT (A / B)
# ─────────────────────────────────────────────────────────────

def load_separate_train(a_path, b_path):
    a = load_image(a_path)
    b = load_image(b_path)
    a, b = random_jitter(a, b)
    a, b = normalize(a, b)
    return a, b


def load_separate_val(a_path, b_path):
    a = load_image(a_path)
    b = load_image(b_path)
    a, b = resize_only(a, b)
    a, b = normalize(a, b)
    return a, b


def make_separate_dataset(folder_a: str, folder_b: str, is_train: bool = True, batch_size: int = BATCH_SIZE):
    """
    Build tf.data dataset from two separate folders.
    Assumes matching filenames in A and B.
    """
    exts = ["*.jpg", "*.jpeg", "*.png"]
    a_files, b_files = [], []

    for ext in exts:
        a_files += sorted(Path(folder_a).glob(ext))
        b_files += sorted(Path(folder_b).glob(ext))

    assert len(a_files) == len(b_files), (
        f"Mismatch: {len(a_files)} files in A vs {len(b_files)} in B"
    )

    a_ds = tf.data.Dataset.from_tensor_slices([str(f) for f in a_files])
    b_ds = tf.data.Dataset.from_tensor_slices([str(f) for f in b_files])
    ds   = tf.data.Dataset.zip((a_ds, b_ds))

    load_fn = load_separate_train if is_train else load_separate_val
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        ds = ds.shuffle(BUFFER_SIZE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────
#  SINGLE IMAGE FOR INFERENCE
# ─────────────────────────────────────────────────────────────

def load_single_image(path: str) -> tf.Tensor:
    """Load, resize, normalize a single image for inference."""
    img = load_image(path)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = (img / 127.5) - 1.0
    img = tf.expand_dims(img, 0)   # Add batch dim
    return img


def tensor_to_image(tensor: tf.Tensor) -> np.ndarray:
    """Convert model output tensor [-1,1] to uint8 [0,255] numpy array."""
    tensor = (tensor + 1.0) * 127.5
    tensor = tf.cast(tensor, tf.uint8)
    return tensor.numpy()
