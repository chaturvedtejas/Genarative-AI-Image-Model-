"""
model.py — Pix2Pix Architecture
────────────────────────────────────────────────────────────────
Generator  : U-Net (encoder-decoder with skip connections)
             Input  → 8 down-sampling blocks with dropout
             Bottleneck → 512 filters
             Output → 8 up-sampling blocks + skip concatenation

Discriminator : PatchGAN (70×70 receptive field)
             Classifies overlapping 70×70 patches as real/fake
             Much more sensitive to local texture than full-image discriminator

Loss:
  G loss = GAN loss (fool discriminator) + λ * L1 loss (pixel-wise)
  D loss = real_loss + fake_loss (binary cross-entropy)

Reference: Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LAMBDA_L1

# ── Weight initializer (as per original paper) ────────────────
init = initializers.RandomNormal(stddev=0.02)


# ─────────────────────────────────────────────────────────────
#  GENERATOR  (U-Net 256)
# ─────────────────────────────────────────────────────────────

def downsample(filters, size, apply_batchnorm=True):
    """Conv → BatchNorm → LeakyReLU  (stride-2 = halves spatial dims)"""
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(
        filters, size, strides=2, padding="same",
        kernel_initializer=init, use_bias=False,
    ))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def upsample(filters, size, apply_dropout=False):
    """TransposedConv → BatchNorm → [Dropout] → ReLU  (stride-2 = doubles spatial dims)"""
    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding="same",
        kernel_initializer=init, use_bias=False,
    ))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block


def build_generator():
    """
    U-Net Generator
    ───────────────
    Encoder (8 downsamples):
      256→128→64→32→16→8→4→2→1  (spatial)
         64  128  256 512 512 512 512 512  (filters)

    Decoder (8 upsamples with skip connections):
      1→2→4→8→16→32→64→128→256
      512+512 → 512+512 → ... → 128+128 → 64+64

    Final: tanh → pixel values in [-1, 1]
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # ── Encoder ────────────────────────────────────────────────
    down_stack = [
        downsample(64,  4, apply_batchnorm=False),  # e1: 256→128
        downsample(128, 4),                          # e2: 128→64
        downsample(256, 4),                          # e3: 64→32
        downsample(512, 4),                          # e4: 32→16
        downsample(512, 4),                          # e5: 16→8
        downsample(512, 4),                          # e6: 8→4
        downsample(512, 4),                          # e7: 4→2
        downsample(512, 4),                          # e8: 2→1  (bottleneck)
    ]

    # ── Decoder ────────────────────────────────────────────────
    up_stack = [
        upsample(512, 4, apply_dropout=True),        # d1: 1→2
        upsample(512, 4, apply_dropout=True),        # d2: 2→4
        upsample(512, 4, apply_dropout=True),        # d3: 4→8
        upsample(512, 4),                            # d4: 8→16
        upsample(256, 4),                            # d5: 16→32
        upsample(128, 4),                            # d6: 32→64
        upsample(64,  4),                            # d7: 64→128
    ]

    last = layers.Conv2DTranspose(
        IMG_CHANNELS, 4, strides=2, padding="same",
        kernel_initializer=init, activation="tanh",  # d8: 128→256
    )

    x = inputs
    skips = []

    # Encode — save skip connections
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])   # Drop bottleneck, reverse for decoder

    # Decode — concatenate skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    return Model(inputs=inputs, outputs=x, name="UNet_Generator")


# ─────────────────────────────────────────────────────────────
#  DISCRIMINATOR  (PatchGAN 70×70)
# ─────────────────────────────────────────────────────────────

def build_discriminator():
    """
    PatchGAN Discriminator
    ──────────────────────
    Classifies each 70×70 patch of the image as real or fake.
    Input: concatenated [source | target] → channels=6

    Architecture:
      Conv(64) → Conv(128) → Conv(256) → Conv(512) → Conv(1)
      Receptive field after 4 strides: ~70×70 pixels
    """
    inp    = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="input_image")
    target = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="target_image")

    x = layers.Concatenate()([inp, target])   # (256, 256, 6)

    x = downsample(64,  4, apply_batchnorm=False)(x)   # 128×128
    x = downsample(128, 4)(x)                           # 64×64
    x = downsample(256, 4)(x)                           # 32×32

    # Zero-pad then stride-1 conv
    x = layers.ZeroPadding2D()(x)                       # 34×34
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)                       # 33×33
    x = layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(x)   # 30×30 patch output

    return Model(inputs=[inp, target], outputs=x, name="PatchGAN_Discriminator")


# ─────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    """
    G_loss = GAN_loss (fool D) + λ * L1_loss (pixel accuracy)

    GAN loss: D should output 1 (real) for generated images
    L1  loss: generated image should be close to ground truth
    """
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss  = tf.reduce_mean(tf.abs(target - gen_output))
    total    = gan_loss + (LAMBDA_L1 * l1_loss)
    return total, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    D_loss = real_loss + fake_loss

    real_loss: D should output 1 for real pairs
    fake_loss: D should output 0 for generated pairs
    """
    real_loss = bce(tf.ones_like(disc_real_output),      disc_real_output)
    fake_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss


# ─────────────────────────────────────────────────────────────
#  MODEL SUMMARY UTIL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G = build_generator()
    D = build_discriminator()
    print("\n── Generator ──────────────────────────────────────")
    G.summary()
    print("\n── Discriminator ──────────────────────────────────")
    D.summary()

    # Quick shape check
    import numpy as np
    dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
    out   = G(dummy, training=False)
    print(f"\n[✓] Generator output shape: {out.shape}")   # (1, 256, 256, 3)
