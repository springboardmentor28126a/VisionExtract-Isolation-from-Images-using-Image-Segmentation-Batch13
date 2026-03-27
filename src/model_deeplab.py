"""
model_deeplab.py  — FIXED VERSION
Place in: src/model_deeplab.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def aspp_block(x, num_filters=128):
    # 1×1 conv
    c1 = layers.Conv2D(num_filters, 1, padding="same", use_bias=False)(x)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)

    # 3×3 dilated conv, rate=6
    c6 = layers.Conv2D(num_filters, 3, padding="same", dilation_rate=6, use_bias=False)(x)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation("relu")(c6)

    # 3×3 dilated conv, rate=12
    c12 = layers.Conv2D(num_filters, 3, padding="same", dilation_rate=12, use_bias=False)(x)
    c12 = layers.BatchNormalization()(c12)
    c12 = layers.Activation("relu")(c12)

    # 3×3 dilated conv, rate=18
    c18 = layers.Conv2D(num_filters, 3, padding="same", dilation_rate=18, use_bias=False)(x)
    c18 = layers.BatchNormalization()(c18)
    c18 = layers.Activation("relu")(c18)

    # Global Average Pooling branch — FIXED: use Resizing layer instead of Lambda
    gap = layers.GlobalAveragePooling2D()(x)
    gap = layers.Reshape((1, 1, x.shape[-1]))(gap)
    gap = layers.Conv2D(num_filters, 1, use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Activation("relu")(gap)
    # Use UpSampling instead of tf.image.resize inside Lambda
    target_h = x.shape[1]
    target_w = x.shape[2]
    gap = layers.Resizing(target_h, target_w)(gap)

    concat = layers.Concatenate()([c1, c6, c12, c18, gap])
    out = layers.Conv2D(num_filters, 1, padding="same", use_bias=False)(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    return out


def build_deeplabv3plus(input_shape=(128, 128, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder: MobileNetV2
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Deep features from block_13
    encoder_output    = mobilenet.get_layer("block_13_expand_relu").output
    # Low-level features from block_3
    low_level_features = mobilenet.get_layer("block_3_expand_relu").output

    # ASPP
    aspp_out = aspp_block(encoder_output, num_filters=128)

    # Upsample ASPP ×4 — FIXED: use Resizing instead of Lambda
    enc_h = encoder_output.shape[1]
    enc_w = encoder_output.shape[2]
    aspp_up = layers.Resizing(enc_h * 4, enc_w * 4)(aspp_out)

    # Reduce low-level channels
    ll = layers.Conv2D(32, 1, padding="same", use_bias=False)(low_level_features)
    ll = layers.BatchNormalization()(ll)
    ll = layers.Activation("relu")(ll)

    # Fuse
    fused = layers.Concatenate()([aspp_up, ll])

    fused = layers.Conv2D(128, 3, padding="same", use_bias=False)(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation("relu")(fused)

    fused = layers.Conv2D(128, 3, padding="same", use_bias=False)(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation("relu")(fused)

    # Upsample to input resolution — FIXED: use Resizing
    up_final = layers.Resizing(input_shape[0], input_shape[1])(fused)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(up_final)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name="VisionExtract_DeepLabV3Plus_MobileNetV2")
    return model, mobilenet


def freeze_encoder(backbone):
    for layer in backbone.layers:
        layer.trainable = False
    frozen = sum(1 for l in backbone.layers if not l.trainable)
    print(f"Encoder FROZEN — {frozen} layers frozen.")


def unfreeze_encoder(backbone, from_block=10):
    unfreeze = False
    unfrozen = 0
    for layer in backbone.layers:
        if f"block_{from_block}" in layer.name:
            unfreeze = True
        if unfreeze:
            layer.trainable = True
            unfrozen += 1
    print(f"Encoder partially UNFROZEN — {unfrozen} layers from block {from_block}.")
    # Add this to utils_deeplab.py

