"""Convolutional U-Net and TransUNet surrogates for 2D wave fields.

Matches the benchmark implementation from keras-unet-collection with native 360x360 resolution
(zero-padded to 368x368, then cropped) and water-masked relative L2 loss.
"""
from typing import Any, Callable, List, Optional, Tuple
import numpy as np


def build_unet_keras(
    input_shape: Tuple[int, int, int] = (360, 360, 4),
    filter_num: List[int] = [64, 128, 256, 512],
    pad: int = 4,
) -> Any:
    """Build U-Net 2D model with zero-padding to 368x368 for divisible pooling.
    Requires TensorFlow and keras-unet-collection.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from keras_unet_collection import models
    except ImportError as e:
        raise ImportError(
            f"TensorFlow or keras-unet-collection not available in current environment: {e}. "
            "Please use the ML environment (/home/model2/ambientes/ML/bin/python3) for Keras U-Net."
        )

    inp = layers.Input(input_shape)
    x = layers.ZeroPadding2D(pad)(inp)

    padded_shape = (input_shape[0] + 2 * pad, input_shape[1] + 2 * pad, input_shape[2])
    u = models.unet_2d(
        padded_shape,
        filter_num,
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        activation="ReLU",
        output_activation=None,
        batch_norm=False,
        pool=True,
        unpool="bilinear",
        name="unet_backbone",
    )
    y = u(x)
    if isinstance(y, (list, tuple)):
        y = y[0]

    out = layers.Cropping2D(pad)(y)
    return tf.keras.Model(inputs=inp, outputs=out, name="unet_360")


def build_transunet_keras(
    input_shape: Tuple[int, int, int] = (360, 360, 4),
    filter_num: List[int] = [64, 128, 256, 512],
    pad: int = 4,
) -> Any:
    """Build TransUNet 2D model with Vision Transformer bottleneck.
    Requires TensorFlow and keras-unet-collection.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from keras_unet_collection import models
    except ImportError as e:
        raise ImportError(f"TensorFlow or keras-unet-collection not available: {e}")

    inp = layers.Input(input_shape)
    x = layers.ZeroPadding2D(pad)(inp)

    padded_shape = (input_shape[0] + 2 * pad, input_shape[1] + 2 * pad, input_shape[2])
    u = models.transunet_2d(
        padded_shape,
        filter_num_skip=filter_num,
        filter_num_aggregate=64,
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        embed_dim=768,
        num_mlp=3072,
        num_heads=12,
        num_transformer=12,
        activation="ReLU",
        output_activation=None,
        batch_norm=False,
        pool=True,
        unpool="bilinear",
        name="transunet_backbone",
    )
    y = u(x)
    if isinstance(y, (list, tuple)):
        y = y[0]

    out = layers.Cropping2D(pad)(y)
    return tf.keras.Model(inputs=inp, outputs=out, name="transunet_360")


def get_keras_masked_rel_l2(mask: np.ndarray, name: str = "rel_l2_inter") -> Callable:
    """TensorFlow/Keras masked relative L2 loss/metric function."""
    import tensorflow as tf

    mm = tf.constant(mask[None, ..., None], dtype=tf.float32)

    def fn(y_true, y_pred):
        num = tf.reduce_sum(tf.square((y_pred - y_true) * mm), axis=[1, 2, 3])
        den = tf.reduce_sum(tf.square(y_true * mm), axis=[1, 2, 3])
        return tf.reduce_mean(tf.sqrt(num / (den + 1e-12)))

    fn.__name__ = name
    return fn
