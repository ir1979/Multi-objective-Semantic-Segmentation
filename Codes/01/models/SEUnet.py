#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Multiply,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import Model

np.random.seed(101)

def SEModule(input_tensor, ratio, out_dim):
    """Squeeze-and-Excitation block with residual shortcut.

    The residual (Add) ensures gradients always flow back through the identity
    path, preventing the stacked SE gates from multiplying signal down to ~0
    (which caused all-background collapse when SE blocks were applied at every
    encoder and decoder stage).
    """
    x = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(units=out_dim // ratio)(x)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)
    scale = Multiply()([input_tensor, excitation])
    return Add()([input_tensor, scale])


def SEUnet(encoder_filters=None, summary=False):
    nClasses=1
    input_height=256
    input_width=256
    f = list(encoder_filters) if encoder_filters is not None else [32, 64, 128, 256, 512]

    def _se_ratio(n):
        return max(2, n // 8)

    inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(f[0],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(f[0],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    # se
    conv1 = SEModule(conv1, _se_ratio(f[0]), f[0])

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(f[1],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(f[1],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    # se
    conv2 = SEModule(conv2, _se_ratio(f[1]), f[1])

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(f[2],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(f[2],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    # se
    conv3 = SEModule(conv3, _se_ratio(f[2]), f[2])

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(f[3],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(f[3],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    # se
    conv4 = SEModule(conv4, _se_ratio(f[3]), f[3])

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(f[4],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(f[4],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # se
    conv5 = SEModule(conv5, _se_ratio(f[4]), f[4])

    up6 = Conv2D(f[3],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    up6 = BatchNormalization()(up6)

    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(f[3],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(f[3],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    # se
    conv6 = SEModule(conv6, _se_ratio(f[3]), f[3])

    up7 = Conv2D(f[2],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)

    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(f[2],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(f[2],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # se
    conv7 = SEModule(conv7, _se_ratio(f[2]), f[2])

    up8 = Conv2D(f[1],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)

    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(f[1],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(f[1],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # se
    conv8 = SEModule(conv8, _se_ratio(f[1]), f[1])

    up9 = Conv2D(f[0],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)

    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(f[0],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(f[0],
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # se
    conv9 = SEModule(conv9, _se_ratio(f[0]), f[0])

    conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    final = Activation('sigmoid')(conv10)
    model = Model(inputs=inputs, outputs=final, name='SEUnet')
    if summary:
        model.summary()

    return model
