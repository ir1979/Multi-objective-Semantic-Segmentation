import numpy as np
from keras import backend as K
from keras.layers import Activation, BatchNormalization, Conv2D, Input, MaxPooling2D, Reshape, UpSampling2D, concatenate
from keras.models import Model
from tensorflow.keras.layers import SpatialDropout2D

np.random.seed(101)

def double_conv_layer(x, size, dropout=0.40, batch_norm=True):
    if K.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def UNET_224():
    dropout_val=0.50
    if K.image_data_format() == 'channels_first':
        inputs = Input((3, 256 , 256))
        axis = 1
    else:
        inputs = Input((256, 256, 3))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val)

    conv_final = Conv2D(1, (1, 1))(up_conv_224)
    pred = Activation('sigmoid')(conv_final)

    model = Model(inputs, pred, name='UNet')
    model.summary()
    
    return model


def build_unet(input_shape=(256, 256, 3), num_classes=1, deep_supervision=False):
    """Build U-Net model for building footprint segmentation.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
    deep_supervision : bool
        Whether to use deep supervision (not supported in this implementation)
    
    Returns
    -------
    tf.keras.Model
        U-Net model
    """
    # This model is fixed at 256x256, so we ignore input_shape for now
    # TODO: Make this configurable for other sizes
    model = UNET_224()
    
    # Adapt output for num_classes if needed
    if num_classes != 1:
        # Add additional output channels if needed
        new_output = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_output')(model.output)
        model = Model(inputs=model.input, outputs=new_output)
    
    return model
