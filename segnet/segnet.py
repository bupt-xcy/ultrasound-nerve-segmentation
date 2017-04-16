from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from skimage.transform import resize

from keras.layers.core import Activation, Layer, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K, models
from keras.callbacks import ModelCheckpoint
import os
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

smooth = 1
img_rows = 96
img_cols = 96

os.environ['CUDA_VISIBLE_DEVICES'] = str(2)


def get_session():
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    # config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(allow_growth=True)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session())

data_path = '/home/chenyangxu/THESIS_EXPERIMENTS/file/'


def load_train_data():
    imgs_train = np.load(data_path + 'train.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load(data_path + 'validation.npy')
    imgs_mask_valid = np.load(data_path + 'validation_mask.npy')
    return imgs_valid, imgs_mask_valid


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_segnet_basic_17():
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
def get_segnet_basic_52():
    kernel = 3

    encoding_layers = [
        Conv2D(64, (3, 3), padding='same', input_shape=(img_rows, img_cols, 1)),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
        # Dropout(0.5),

        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
        # Dropout(0.5),

        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
        # Dropout(0.5),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
        # Dropout(0.5),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
        # Dropout(0.5),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(size=(2,2)),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        # Dropout(0.5),

        UpSampling2D(size=(2,2)),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        # Dropout(0.5),

        UpSampling2D(size=(2,2)),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        # Dropout(0.5),

        UpSampling2D(size=(2,2)),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        # Dropout(0.5),

        UpSampling2D(size=(2,2)),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        # Dropout(0.5),

        Conv2D(1, (1, 1), padding='valid'),
        BatchNormalization(axis=3),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Activation('sigmoid'))
    # autoencoder.add(Conv2D(1, (1, 1), padding='valid', activation='sigmoid'))

    # optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
    autoencoder.compile(loss=dice_coef_loss, optimizer=Adam(lr=1e-3), metrics=[dice_coef])
    # autoencoder.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice_coef])

    return autoencoder


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_valid, imgs_mask_valid = load_validation_data()

    imgs_train = preprocess(imgs_train)
    print(imgs_train.shape)
    imgs_mask_train = preprocess(imgs_mask_train)
    print(imgs_mask_train.shape)
    imgs_valid = preprocess(imgs_valid)
    print(imgs_valid.shape)
    imgs_mask_valid = preprocess(imgs_mask_valid)
    print(imgs_mask_valid.shape)

    imgs_train = imgs_train.astype('float32')
    imgs_valid = imgs_valid.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    val_mean = np.mean(imgs_valid)
    val_std = np.std(imgs_valid)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid -= val_mean
    imgs_valid /= val_std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_valid = imgs_mask_valid.astype('float32')
    imgs_mask_valid /= 255.

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_segnet_basic_52()
    model_checkpoint = ModelCheckpoint('/home/chenyangxu/THESIS_EXPERIMENTS/segnet.hdf5', monitor='val_loss',
                                       save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    his = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=100, verbose=1, shuffle=True,
                    validation_data=(imgs_valid, imgs_mask_valid), callbacks=[model_checkpoint])
    his_loss = his.history["loss"]
    val_loss = his.history["val_loss"]
    numpy_loss_history = np.array(his_loss)
    numpy_val_loss_history = np.array(val_loss)
    np.savetxt("/home/chenyangxu/THESIS_EXPERIMENTS/segnet_logs/loss.txt", numpy_loss_history, delimiter=",")
    np.savetxt("/home/chenyangxu/THESIS_EXPERIMENTS/segnet_logs/val_loss.txt", numpy_val_loss_history, delimiter=",")


if __name__ == '__main__':
    train()
    # model = get_segnet_basic()
    # print (model.summary())
