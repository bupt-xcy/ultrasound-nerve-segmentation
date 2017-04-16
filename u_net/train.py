from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF

from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.

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


# data_path = '/Users/xuchenyang/Documents/sec_exp/file/'


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
    model = get_unet()
    model_checkpoint = ModelCheckpoint('/home/chenyangxu/THESIS_EXPERIMENTS/unet.hdf5', monitor='val_loss',
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
    np.savetxt("/home/chenyangxu/THESIS_EXPERIMENTS/logs/loss.txt", numpy_loss_history, delimiter=",")
    np.savetxt("/home/chenyangxu/THESIS_EXPERIMENTS/logs/val_loss.txt", numpy_val_loss_history, delimiter=",")


if __name__ == '__main__':
    # model = get_unet()
    # print(model.summary())
    train()
