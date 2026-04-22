"""Example custom models registered through the plugin system."""

from tensorflow import keras

from models import register_model_builder


def build_tiny_paper_unet(**kwargs):
    """Small baseline example showing how to plug in a custom model."""
    filters = int(kwargs.get("filters", 16))
    inputs = keras.Input(shape=kwargs["input_shape"])
    x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    outputs = keras.layers.Conv2D(kwargs["num_classes"], 1, activation=kwargs.get("activation", "sigmoid"))(x)
    return keras.Model(inputs, outputs, name="TinyPaperUNet")


register_model_builder(
    name="TinyPaperUNet",
    builder=build_tiny_paper_unet,
    aliases=["tiny_paper_unet"],
    metadata={"family": "plugin_example"},
    overwrite=True,
)
