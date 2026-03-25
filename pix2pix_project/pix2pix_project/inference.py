"""
inference.py — Run trained Pix2Pix generator on new images
────────────────────────────────────────────────────────────────
Features:
  • Load saved generator (.h5 or checkpoint)
  • Auto-generate edge/sketch from real photo using Canny + HED-style processing
  • Batch inference on folder
  • Side-by-side comparison output
"""

import os, cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from config import GENERATOR_PATH, IMG_HEIGHT, IMG_WIDTH
from dataset import load_single_image, tensor_to_image


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

_generator = None   # Module-level singleton

def get_generator(model_path: str = GENERATOR_PATH):
    global _generator
    if _generator is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Generator not found at {model_path}. Run train.py first."
            )
        print(f"[Inference] Loading generator from {model_path} …")
        _generator = tf.keras.models.load_model(model_path, compile=False)
        print("[Inference] Ready ✓")
    return _generator


# ─────────────────────────────────────────────────────────────
#  EDGE / SKETCH EXTRACTION  (for sketch→photo mode)
# ─────────────────────────────────────────────────────────────

def extract_edges(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Convert a real photo to an edge map suitable as sketch input.
    Uses multi-scale Canny with Gaussian blur for clean edges.
    """
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserves edges while smoothing texture noise
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Canny — auto-compute thresholds from median
    med = np.median(filtered)
    lo  = int(max(0,   0.66 * med))
    hi  = int(min(255, 1.33 * med))
    edges = cv2.Canny(filtered, lo, hi)

    # Dilate slightly for thicker, more sketch-like lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Invert: white background, black lines (typical sketch style)
    edges = cv2.bitwise_not(edges)

    # Convert to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if output_path:
        cv2.imwrite(output_path, edges_3ch)

    return edges_3ch


def preprocess_for_inference(image: np.ndarray) -> tf.Tensor:
    """Resize a BGR numpy image and convert to model input tensor."""
    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
    t    = tf.cast(rgb, tf.float32)
    t    = (t / 127.5) - 1.0
    return tf.expand_dims(t, 0)


# ─────────────────────────────────────────────────────────────
#  INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def translate_image(input_path: str, output_path: str, auto_edge: bool = False) -> dict:
    """
    Translate a single image.

    Args:
        input_path  : path to source image (sketch, day photo, etc.)
        output_path : where to save the generated image
        auto_edge   : if True, auto-extract edges from a real photo first

    Returns:
        dict with paths + metrics
    """
    generator = get_generator()

    if auto_edge:
        # Convert photo → edge map first
        edge_img = extract_edges(input_path)
        inp_tensor = preprocess_for_inference(edge_img)
    else:
        inp_tensor = load_single_image(input_path)

    # Run generator
    generated = generator(inp_tensor, training=False)
    output_arr = tensor_to_image(generated[0])   # (256, 256, 3)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(output_arr).save(output_path)

    return {
        "input_path":  input_path,
        "output_path": output_path,
        "shape":       output_arr.shape,
    }


def translate_with_comparison(input_path: str, output_path: str, auto_edge: bool = False) -> str:
    """
    Generate a side-by-side comparison image:
    [Input | Generated]
    """
    import matplotlib.pyplot as plt

    generator  = get_generator()

    if auto_edge:
        edge_img   = extract_edges(input_path)
        inp_tensor = preprocess_for_inference(edge_img)
        display_in = cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB)
    else:
        inp_tensor = load_single_image(input_path)
        orig = cv2.imread(input_path)
        display_in = cv2.cvtColor(
            cv2.resize(orig, (IMG_WIDTH, IMG_HEIGHT)), cv2.COLOR_BGR2RGB
        )

    generated  = generator(inp_tensor, training=False)
    output_arr = tensor_to_image(generated[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_in);  axes[0].set_title("Input");     axes[0].axis("off")
    axes[1].imshow(output_arr);  axes[1].set_title("Generated"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def batch_translate(input_folder: str, output_folder: str, auto_edge: bool = False):
    """Translate all images in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in exts]

    print(f"[Batch] Translating {len(files)} images …")
    for fname in files:
        inp  = os.path.join(input_folder, fname)
        out  = os.path.join(output_folder, fname)
        translate_image(inp, out, auto_edge=auto_edge)
        print(f"  [✓] {fname}")

    print(f"[✓] Batch complete → {output_folder}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Input image or folder")
    parser.add_argument("--output",  required=True, help="Output image or folder")
    parser.add_argument("--compare", action="store_true", help="Save side-by-side comparison")
    parser.add_argument("--auto_edge", action="store_true", help="Extract edges from photo first")
    parser.add_argument("--batch",   action="store_true", help="Process an entire folder")
    args = parser.parse_args()

    if args.batch:
        batch_translate(args.input, args.output, auto_edge=args.auto_edge)
    elif args.compare:
        translate_with_comparison(args.input, args.output, auto_edge=args.auto_edge)
    else:
        translate_image(args.input, args.output, auto_edge=args.auto_edge)
