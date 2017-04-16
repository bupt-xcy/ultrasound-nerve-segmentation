from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF

from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K, models

img_rows = 96
img_cols = 96

smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def get_unet():
    inputs = Input(shape=(img_rows, img_cols, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    BatchNormalization(axis=3)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    BatchNormalization(axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    BatchNormalization(axis=3)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    # BatchNormalization(axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    # BatchNormalization(axis=3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # BatchNormalization(axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    # BatchNormalization(axis=3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # BatchNormalization(axis=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # BatchNormalization(axis=3)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    # BatchNormalization(axis=3)
    # conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    # BatchNormalization(axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    # BatchNormalization(axis=3)
    # conv6 = Dropout(0.5)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    # BatchNormalization(axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # BatchNormalization(axis=3)
    # conv7 = Dropout(0.5)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    # BatchNormalization(axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # BatchNormalization(axis=3)
    # conv8 = Dropout(0.5)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    # BatchNormalization(axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    # BatchNormalization(axis=3)
    # conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    # BatchNormalization(axis=3)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def get_segnet_basic():
    kernel = 3

    encoding_layers = [
        # Conv2D(64, (3, 3), padding='same', input_shape=(img_rows, img_cols, 1)),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(64, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # MaxPooling2D(),
        # # Dropout(0.5),
        #
        # Conv2D(128, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(128, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # MaxPooling2D(),
        # # Dropout(0.5),
        #
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # MaxPooling2D(),
        # # Dropout(0.5),
        #
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # MaxPooling2D(),
        # # Dropout(0.5),
        #
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # MaxPooling2D(),
        # # Dropout(0.5),

        ZeroPadding2D(padding=(1, 1), input_shape=(img_rows, img_cols, 1)),
        Conv2D(64, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(128, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(256, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(512, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        # UpSampling2D(size=(2,2)),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # # Dropout(0.5),
        #
        # UpSampling2D(size=(2,2)),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(512, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # # Dropout(0.5),
        #
        # UpSampling2D(size=(2,2)),
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(256, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(128, (kernel, kernel), padding='same'),
        # BatchNormalization(),
        # Activation('relu'),
        # # Dropout(0.5),
        #
        # UpSampling2D(size=(2,2)),
        # Conv2D(128, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # Conv2D(64, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # # Dropout(0.5),
        #
        # UpSampling2D(size=(2,2)),
        # Conv2D(64, (kernel, kernel), padding='same'),
        # BatchNormalization(axis=3),
        # Activation('relu'),
        # # Dropout(0.5),
        #
        # Conv2D(1, (1, 1), padding='valid'),
        # BatchNormalization(axis=3),
        UpSampling2D(size=(2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(512, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),

        UpSampling2D(size=(2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(256, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),

        UpSampling2D(size=(2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(128, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),

        UpSampling2D(size=(2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(64, (kernel, kernel), padding='valid'),
        BatchNormalization(axis=3),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    # autoencoder.add(Activation('sigmoid'))
    autoencoder.add(Conv2D(1, (1, 1), padding='valid', activation='sigmoid'))
    # autoencoder.add(Conv2D(1, (1, 1), padding='valid', activation='sigmoid'))

    # optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
    autoencoder.compile(loss=dice_coef_loss, optimizer=Adam(lr=1e-2), metrics=[dice_coef])
    # autoencoder.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice_coef])

    return autoencoder