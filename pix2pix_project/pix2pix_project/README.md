# Pix2Pix — Image-to-Image Translation Studio
### Final Year Project · Conditional GAN · U-Net + PatchGAN

---

## What This Project Does

A full end-to-end **Conditional GAN** that learns to translate images from one domain to another:

| Mode | Input | Output |
|---|---|---|
| `sketch2photo` | Hand-drawn sketch or edge map | Realistic photo |
| `day2night` | Daytime scene | Night scene |
| `edges2shoes` | Edge map | Photorealistic shoe |
| `facades` | Building label map | Facade photo |

Comes with a beautiful **web app** where you can upload an image (or draw one!) and get the translated result instantly.

---

## Architecture

```
Generator  : U-Net (256×256)
  Encoder  → 8× Conv stride-2  (64→128→256→512→512→512→512→512 filters)
  Decoder  → 8× TransposedConv + skip connections from encoder
  Output   → tanh activation → pixel values [-1, 1]

Discriminator : PatchGAN (70×70 receptive field)
  Input    → concatenated [source | generated]  (6 channels)
  Output   → 30×30 grid of real/fake scores
  Advantage → sensitive to local texture, not just global structure

Loss:
  G_loss = BCE(D(x, G(x)), 1) + λ * L1(G(x), y)   λ=100
  D_loss = BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)
```

---

## Project Structure

```
pix2pix_project/
├── config.py               ← All hyperparameters
├── model.py                ← U-Net Generator + PatchGAN Discriminator
├── dataset.py              ← tf.data pipeline + augmentation
├── train.py                ← Full training loop (checkpoints, metrics, TensorBoard)
├── inference.py            ← Run trained model on new images
├── app.py                  ← Flask web server
├── requirements.txt
├── notebooks/
│   └── Train_Pix2Pix_Colab.ipynb  ← Full Colab training notebook
├── templates/
│   └── index.html          ← Web app (upload/draw + translate)
├── models/                 ← Saved generator.h5 + checkpoints
├── dataset/
│   ├── train/              ← Training images (paired or A/B folders)
│   └── val/
├── static/
│   ├── uploads/            ← User uploads
│   └── results/            ← Generated outputs
└── logs/                   ← TensorBoard logs + training curves
```

---

## Quick Start

### Option A — Train on Google Colab (recommended)
1. Open `notebooks/Train_Pix2Pix_Colab.ipynb` in Colab
2. Set runtime to **GPU (T4)**
3. Run all cells — trains in ~2-3 hours
4. Download `generator.h5` to `models/`

### Option B — Train locally
```bash
pip install -r requirements.txt
python train.py --epochs 150 --mode paired
```

### Run the web app
```bash
python app.py
# Open http://localhost:5000
```

---

## Datasets

All free and publicly available:

| Dataset | Source | Size |
|---|---|---|
| Facades | Berkeley Pix2Pix | 400 pairs |
| Edges2Shoes | Berkeley Pix2Pix | 50K pairs |
| Day2Night | Kaggle | 17K pairs |
| CMP Facades | Kaggle | 606 pairs |

Download the Berkeley datasets automatically:
```bash
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
```

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **SSIM** | Structural similarity (0→1) | > 0.75 |
| **PSNR** | Peak signal-to-noise ratio (dB) | > 22 dB |
| **FID** | Fréchet Inception Distance (lower=better) | < 50 |
| **L1 Loss** | Pixel-wise reconstruction loss | Minimize |

Monitor live in TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## Web App Features

- **Upload mode** — drop any image and translate it
- **Draw mode** — sketch directly in the browser, translate your drawing
- **Auto Edge mode** — upload a real photo, auto-extract edges, then translate
- **History** — browse all previous translations in session
- **Download** — save any generated image
- **Compare** — toggle between input and output

---

## Research Angles (for paper/report)

1. **L1 weight ablation** — compare λ=1 vs λ=10 vs λ=100 on SSIM
2. **Batch size effect** — Pix2Pix uses batch=1 (instance norm); compare with batch=4
3. **Generator depth** — U-Net-64 vs U-Net-256 quality/speed tradeoff
4. **PatchGAN receptive field** — 16×16 vs 70×70 vs full-image discriminator
5. **Domain transfer** — train on facades, fine-tune on day2night (few-shot)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras |
| Generator | U-Net (Isola et al. 2017) |
| Discriminator | PatchGAN 70×70 |
| Training | @tf.function, AdamW, L1+GAN loss |
| Backend | Flask |
| Frontend | Vanilla JS + Canvas API |
| Metrics | SSIM, PSNR (tf.image) |

---

## Reference

Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017).
*Image-to-image translation with conditional adversarial networks.*
CVPR 2017. https://arxiv.org/abs/1611.07004

---

*Final Year Project — Computer Science / Artificial Intelligence*
