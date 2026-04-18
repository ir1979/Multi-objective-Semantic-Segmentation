import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, name=None):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=None if name is None else f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=None if name is None else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu2")(x)
    return x


def build_unet_pp(input_shape=(256, 256, 3), num_classes=1, base_filters=32, deep_supervision=False):
    """Build a UNet++ model with optional deep supervision."""
    nb_filter = [
        base_filters,
        base_filters * 2,
        base_filters * 4,
        base_filters * 8,
        base_filters * 16,
    ]

    inputs = layers.Input(shape=input_shape)

    x0_0 = conv_block(inputs, nb_filter[0], name="x0_0")
    x1_0 = conv_block(layers.MaxPool2D(pool_size=2)(x0_0), nb_filter[1], name="x1_0")
    x2_0 = conv_block(layers.MaxPool2D(pool_size=2)(x1_0), nb_filter[2], name="x2_0")
    x3_0 = conv_block(layers.MaxPool2D(pool_size=2)(x2_0), nb_filter[3], name="x3_0")
    x4_0 = conv_block(layers.MaxPool2D(pool_size=2)(x3_0), nb_filter[4], name="x4_0")

    x0_1 = conv_block(
        layers.concatenate([
            x0_0,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x1_0),
        ], axis=-1),
        nb_filter[0],
        name="x0_1",
    )

    x1_1 = conv_block(
        layers.concatenate([
            x1_0,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x2_0),
        ], axis=-1),
        nb_filter[1],
        name="x1_1",
    )

    x0_2 = conv_block(
        layers.concatenate([
            x0_0,
            x0_1,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x1_1),
        ], axis=-1),
        nb_filter[0],
        name="x0_2",
    )

    x2_1 = conv_block(
        layers.concatenate([
            x2_0,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x3_0),
        ], axis=-1),
        nb_filter[2],
        name="x2_1",
    )

    x1_2 = conv_block(
        layers.concatenate([
            x1_0,
            x1_1,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x2_1),
        ], axis=-1),
        nb_filter[1],
        name="x1_2",
    )

    x0_3 = conv_block(
        layers.concatenate([
            x0_0,
            x0_1,
            x0_2,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x1_2),
        ], axis=-1),
        nb_filter[0],
        name="x0_3",
    )

    x3_1 = conv_block(
        layers.concatenate([
            x3_0,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x4_0),
        ], axis=-1),
        nb_filter[3],
        name="x3_1",
    )

    x2_2 = conv_block(
        layers.concatenate([
            x2_0,
            x2_1,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x3_1),
        ], axis=-1),
        nb_filter[2],
        name="x2_2",
    )

    x1_3 = conv_block(
        layers.concatenate([
            x1_0,
            x1_1,
            x1_2,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x2_2),
        ], axis=-1),
        nb_filter[1],
        name="x1_3",
    )

    x0_4 = conv_block(
        layers.concatenate([
            x0_0,
            x0_1,
            x0_2,
            x0_3,
            layers.UpSampling2D(size=2, interpolation="bilinear")(x1_3),
        ], axis=-1),
        nb_filter[0],
        name="x0_4",
    )

    if deep_supervision:
        out1 = layers.Conv2D(num_classes, 1, activation="sigmoid", name="output_1")(x0_1)
        out2 = layers.Conv2D(num_classes, 1, activation="sigmoid", name="output_2")(x0_2)
        out3 = layers.Conv2D(num_classes, 1, activation="sigmoid", name="output_3")(x0_3)
        out4 = layers.Conv2D(num_classes, 1, activation="sigmoid", name="output_4")(x0_4)
        model = Model(inputs, [out1, out2, out3, out4], name="UNetPP_deep_supervision")
    else:
        out = layers.Conv2D(num_classes, 1, activation="sigmoid", name="output")(x0_4)
        model = Model(inputs, out, name="UNetPP")

    return model
