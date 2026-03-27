import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(g, s, num_filters):
    """
    g: Gating signal from decoder
    s: Skip connection from encoder
    """
    wg = layers.Conv2D(num_filters, 1, padding="same")(g)
    wg = layers.BatchNormalization()(wg)
    
    ws = layers.Conv2D(num_filters, 1, padding="same")(s)
    ws = layers.BatchNormalization()(ws)
    
    out = layers.Activation("relu")(layers.add([wg, ws]))
    out = layers.Conv2D(1, 1, padding="same")(out)
    out = layers.Activation("sigmoid")(out)
    
    return layers.multiply([s, out])

def build_unet_v2(input_shape):
    inputs = layers.Input(input_shape)

    # --- Encoder ---
    s1 = conv_block(inputs, 64)
    p1 = layers.MaxPool2D((2, 2))(s1)

    s2 = conv_block(p1, 128)
    p2 = layers.MaxPool2D((2, 2))(s2)

    s3 = conv_block(p2, 256)
    p3 = layers.MaxPool2D((2, 2))(s3)

    s4 = conv_block(p3, 512)
    p4 = layers.MaxPool2D((2, 2))(s4)

    # --- Bridge (Bottleneck) with Dropout ---
    b1 = conv_block(p4, 1024)
    b1 = layers.Dropout(0.5)(b1) # Re-added Dropout for Week 5 [cite: 46]

    # --- Decoder with Attention Gates ---
    u1 = layers.Conv2DTranspose(512, (2, 2), strides=2, padding="same")(b1)
    a1 = attention_gate(u1, s4, 512) 
    d1 = layers.Concatenate()([u1, a1])
    d1 = conv_block(d1, 512)

    u2 = layers.Conv2DTranspose(256, (2, 2), strides=2, padding="same")(d1)
    a2 = attention_gate(u2, s3, 256)
    d2 = layers.Concatenate()([u2, a2])
    d2 = conv_block(d2, 256)

    u3 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding="same")(d2)
    a3 = attention_gate(u3, s2, 128)
    d3 = layers.Concatenate()([u3, a3])
    d3 = conv_block(d3, 128)

    u4 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding="same")(d3)
    a4 = attention_gate(u4, s1, 64)
    d4 = layers.Concatenate()([u4, a4])
    d4 = conv_block(d4, 64)

    # Output Layer for Binary Segmentation [cite: 40, 58]
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    return models.Model(inputs, outputs, name="VisionExtract_Attention_UNet_v2")