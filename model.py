"""
model.py - Convolutional Autoencoder Architecture for Surface Defect Detection.

Theoretical Overview:
--------------------
An Autoencoder is an unsupervised neural network trained to reconstruct its own input.
For anomaly detection, the autoencoder is trained exclusively on defect-free ('normal')
images of manufactured products.

1. Encoder:
   Compacts the high-dimensional input image (256x256x3 = 196,608 values) through consecutive
   strided 2D convolutions and non-linearities (ReLU) down to a compact bottleneck representation
   (8x8x64 = 4,096 values, a 48x compression factor).
   This forces the latent representation to capture only dominant, repetitive structural features
   and ignore high-frequency variations.

2. Decoder:
   Reconstructs the original image from the compressed latent vector using transposed convolutions
   (Conv2DTranspose). The final layer uses a Sigmoid activation function to constrain output pixel
   intensities to [0.0, 1.0], matching the normalized input images.

3. Inference Principle:
   When a normal image is passed, the autoencoder reconstructs it with very low error (MSE ~ 0).
   When an image containing an unseen defect (scratch, crack, hole) is passed, the autoencoder
   fails to reconstruct the abnormal pattern (having never seen defects during training).
   The pixel-wise residual:
       E(x, y) = |I(x, y) - I_hat(x, y)|
   highlights the defect location as a high-error anomaly.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def combined_mse_ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.8) -> tf.Tensor:
    """
    Combined Mean Squared Error (MSE) and Structural Similarity (SSIM) loss:
        Loss = alpha * MSE + (1 - alpha) * (1 - SSIM)

    MSE encourages global pixel intensity match, while SSIM enforces structural,
    edge, and perceptual texture fidelity.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    # SSIM expects values in range [0, 1]
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return alpha * mse + (1.0 - alpha) * (1.0 - ssim)


def build_autoencoder(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    latent_dim_channels: int = 64,
    learning_rate: float = 1e-3,
    loss_type: str = "mse"
) -> models.Model:
    """
    Builds and compiles the Convolutional Autoencoder.

    Args:
        input_shape: Shape of the input images, defaults to (256, 256, 3).
        latent_dim_channels: Number of channels in the 8x8 latent bottleneck.
        learning_rate: Adam optimizer learning rate.
        loss_type: 'mse' or 'combined' (MSE + SSIM).

    Returns:
        Compiled Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ==========================
    # ENCODER (Downsampling)
    # ==========================
    # Block 1: 256x256x3 -> 128x128x32
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.ReLU(name="enc_relu1")(x)

    # Block 2: 128x128x32 -> 64x64x64
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.ReLU(name="enc_relu2")(x)

    # Block 3: 64x64x64 -> 32x32x128
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.ReLU(name="enc_relu3")(x)

    # Block 4: 32x32x128 -> 16x16x256
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same", name="enc_conv4")(x)
    x = layers.BatchNormalization(name="enc_bn4")(x)
    x = layers.ReLU(name="enc_relu4")(x)

    # Bottleneck Latent Layer: 16x16x256 -> 8x8x64
    bottleneck = layers.Conv2D(latent_dim_channels, kernel_size=3, strides=2, padding="same", name="latent_bottleneck")(x)
    bottleneck = layers.BatchNormalization(name="bottleneck_bn")(bottleneck)
    latent = layers.ReLU(name="latent_features")(bottleneck)

    # ==========================
    # DECODER (Upsampling)
    # ==========================
    # Block 1: 8x8x64 -> 16x16x256
    d = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", name="dec_deconv1")(latent)
    d = layers.BatchNormalization(name="dec_bn1")(d)
    d = layers.ReLU(name="dec_relu1")(d)

    # Block 2: 16x16x256 -> 32x32x128
    d = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", name="dec_deconv2")(d)
    d = layers.BatchNormalization(name="dec_bn2")(d)
    d = layers.ReLU(name="dec_relu2")(d)

    # Block 3: 32x32x128 -> 64x64x64
    d = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", name="dec_deconv3")(d)
    d = layers.BatchNormalization(name="dec_bn3")(d)
    d = layers.ReLU(name="dec_relu3")(d)

    # Block 4: 64x64x64 -> 128x128x32
    d = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", name="dec_deconv4")(d)
    d = layers.BatchNormalization(name="dec_bn4")(d)
    d = layers.ReLU(name="dec_relu4")(d)

    # Output Block: 128x128x32 -> 256x256x3
    # Sigmoid activation guarantees pixel outputs in [0.0, 1.0]
    outputs = layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid", name="reconstruction_output")(d)

    autoencoder = models.Model(inputs=inputs, outputs=outputs, name="IndustrialDefectAutoencoder")

    # Select loss function
    if loss_type.lower() == "combined":
        chosen_loss = combined_mse_ssim_loss
    else:
        chosen_loss = "mse"

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss=chosen_loss,
        metrics=["mse", "mae"]
    )
    return autoencoder


def print_architecture_summary():
    """Builds a temporary model instance and prints its summary table."""
    model = build_autoencoder()
    model.summary()
    return model


if __name__ == "__main__":
    print("Testing Autoencoder construction:")
    model = print_architecture_summary()
    print("\nModel successfully built and compiled!")
