"""
train.py — Pix2Pix Training Loop
────────────────────────────────────────────────────────────────
Features:
  • @tf.function compiled training step (fast)
  • TensorBoard logging (G loss, D loss, L1, FID approximation)
  • Checkpoint save/restore
  • Sample image grid every N epochs
  • SSIM + PSNR metrics on validation set
  • Colab-friendly progress bars via tqdm

Usage:
  python train.py --dataset facades --epochs 150 --mode paired
  python train.py --dataset day2night --epochs 150 --mode separate
"""

import os, time, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from model   import build_generator, build_discriminator, generator_loss, discriminator_loss
from dataset import make_paired_dataset, make_separate_dataset, tensor_to_image
from config  import *


# ─────────────────────────────────────────────────────────────
#  OPTIMIZERS
# ─────────────────────────────────────────────────────────────

generator_optimizer     = tf.keras.optimizers.Adam(GEN_LR,  beta_1=BETA_1, beta_2=BETA_2)
discriminator_optimizer = tf.keras.optimizers.Adam(DISC_LR, beta_1=BETA_1, beta_2=BETA_2)


# ─────────────────────────────────────────────────────────────
#  TRAINING STEP
# ─────────────────────────────────────────────────────────────

@tf.function
def train_step(generator, discriminator, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator: source → fake target
        gen_output = generator(input_image, training=True)

        # Discriminator: judge real and fake pairs
        disc_real_output      = discriminator([input_image, target],     training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Losses
        gen_total, gen_gan, gen_l1 = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Apply gradients
    gen_grads  = gen_tape.gradient(gen_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(    zip(gen_grads,  generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total, gen_gan, gen_l1, disc_loss


# ─────────────────────────────────────────────────────────────
#  SAMPLE IMAGE GRID
# ─────────────────────────────────────────────────────────────

def generate_samples(generator, val_dataset, epoch: int, n_samples: int = 4):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    axes[0, 0].set_title("Input",    fontsize=14, pad=10)
    axes[0, 1].set_title("Generated",fontsize=14, pad=10)
    axes[0, 2].set_title("Target",   fontsize=14, pad=10)

    for i, (inp, tar) in enumerate(val_dataset.take(n_samples)):
        pred = generator(inp, training=False)
        axes[i, 0].imshow(tensor_to_image(inp[0]))
        axes[i, 1].imshow(tensor_to_image(pred[0]))
        axes[i, 2].imshow(tensor_to_image(tar[0]))
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  [✓] Samples saved → {path}")


# ─────────────────────────────────────────────────────────────
#  VALIDATION METRICS  (SSIM + PSNR)
# ─────────────────────────────────────────────────────────────

def compute_metrics(generator, val_dataset, n_batches: int = 20):
    ssim_scores, psnr_scores = [], []

    for inp, tar in val_dataset.take(n_batches):
        pred = generator(inp, training=False)
        # Convert [-1,1] → [0,1] for metrics
        pred_01 = (pred + 1.0) / 2.0
        tar_01  = (tar  + 1.0) / 2.0

        ssim = tf.image.ssim(tar_01, pred_01, max_val=1.0)
        psnr = tf.image.psnr(tar_01, pred_01, max_val=1.0)

        ssim_scores.append(float(tf.reduce_mean(ssim)))
        psnr_scores.append(float(tf.reduce_mean(psnr)))

    return np.mean(ssim_scores), np.mean(psnr_scores)


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(args):
    print("=" * 62)
    print("  Pix2Pix Image-to-Image Translation — Training")
    print("=" * 62)

    # GPU memory growth
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Build models ─────────────────────────────────────────
    generator     = build_generator()
    discriminator = build_discriminator()
    print(f"[✓] Generator params     : {generator.count_params():,}")
    print(f"[✓] Discriminator params : {discriminator.count_params():,}")

    # ── Dataset ──────────────────────────────────────────────
    if args.mode == "paired":
        train_ds = make_paired_dataset(os.path.join(DATASET_PATH, "train"), is_train=True)
        val_ds   = make_paired_dataset(os.path.join(DATASET_PATH, "val"),   is_train=False)
    else:
        train_ds = make_separate_dataset(TRAIN_A, TRAIN_B, is_train=True)
        val_ds   = make_separate_dataset(VAL_A,   VAL_B,   is_train=False)

    # ── Checkpointing ─────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, CHECKPOINT_DIR, max_to_keep=3
    )
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"[✓] Restored checkpoint: {ckpt_manager.latest_checkpoint}")

    # ── TensorBoard ──────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_DIR)

    # ── History for plots ─────────────────────────────────────
    history = {"gen_total":[], "gen_gan":[], "gen_l1":[], "disc":[], "ssim":[], "psnr":[]}

    # ── Training ─────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        gen_total_avg = gen_gan_avg = gen_l1_avg = disc_avg = 0.0
        steps = 0

        pbar = tqdm(train_ds, desc=f"Epoch {epoch:03d}/{args.epochs}", unit="step", leave=False)
        for inp, tar in pbar:
            g_tot, g_gan, g_l1, d_loss = train_step(generator, discriminator, inp, tar)
            gen_total_avg += g_tot; gen_gan_avg += g_gan
            gen_l1_avg    += g_l1; disc_avg    += d_loss
            steps += 1
            pbar.set_postfix(G=f"{g_tot:.3f}", D=f"{d_loss:.3f}", L1=f"{g_l1:.4f}")

        # Epoch averages
        gen_total_avg /= steps; gen_gan_avg /= steps
        gen_l1_avg    /= steps; disc_avg    /= steps

        # SSIM / PSNR every 5 epochs
        ssim = psnr = 0.0
        if epoch % 5 == 0:
            ssim, psnr = compute_metrics(generator, val_ds)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"G={gen_total_avg:.4f}  D={disc_avg:.4f}  "
            f"L1={gen_l1_avg:.4f}  "
            f"SSIM={ssim:.3f}  PSNR={psnr:.2f}  "
            f"({elapsed:.0f}s)"
        )

        # TensorBoard logging
        with summary_writer.as_default():
            tf.summary.scalar("G_total", gen_total_avg, step=epoch)
            tf.summary.scalar("G_gan",   gen_gan_avg,   step=epoch)
            tf.summary.scalar("G_L1",    gen_l1_avg,    step=epoch)
            tf.summary.scalar("D_loss",  disc_avg,      step=epoch)
            if ssim: tf.summary.scalar("SSIM", ssim, step=epoch)
            if psnr: tf.summary.scalar("PSNR", psnr, step=epoch)

        # Save history
        history["gen_total"].append(float(gen_total_avg))
        history["gen_gan"].append(float(gen_gan_avg))
        history["gen_l1"].append(float(gen_l1_avg))
        history["disc"].append(float(disc_avg))
        history["ssim"].append(ssim)
        history["psnr"].append(psnr)

        # Generate samples
        if epoch % SAMPLE_FREQ == 0:
            generate_samples(generator, val_ds, epoch)

        # Save checkpoint
        if epoch % CHECKPOINT_FREQ == 0:
            ckpt_manager.save()
            print(f"  [✓] Checkpoint saved (epoch {epoch})")

    # ── Save final generator ──────────────────────────────────
    generator.save(GENERATOR_PATH)
    print(f"\n[✓] Generator saved → {GENERATOR_PATH}")

    # ── Final training curves ─────────────────────────────────
    _plot_history(history, args.epochs)
    return generator


def _plot_history(history, epochs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(1, epochs + 1)

    axes[0].plot(x, history["gen_total"], label="G total")
    axes[0].plot(x, history["disc"],      label="D loss")
    axes[0].set_title("GAN Losses"); axes[0].legend()

    axes[1].plot(x, history["gen_l1"], color="orange", label="L1 loss")
    axes[1].set_title("L1 Loss (Reconstruction)"); axes[1].legend()

    ssim_vals = [v for v in history["ssim"] if v > 0]
    if ssim_vals:
        axes[2].plot([i*5 for i in range(len(ssim_vals))], ssim_vals, label="SSIM", color="green")
        axes[2].set_title("SSIM (Validation)"); axes[2].legend()

    plt.tight_layout()
    out = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[✓] Training curves → {out}")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--dataset", type=str, default="facades")
    parser.add_argument("--mode",    type=str, default="paired",
                        choices=["paired", "separate"],
                        help="paired=side-by-side images, separate=A/B folders")
    args = parser.parse_args()
    train(args)
