from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.io import imsave, imread
import os
import cv2
#
# data_path = "/Users/xuchenyang/Documents/third_exp/train/"
# imgs_mask_train = np.load(data_path + 'train.npy')
# print(imgs_mask_train.shape[0])

img = imread("/Users/xuchenyang/Downloads/train/1_9.tif")
# img = imread("/Users/xuchenyang/Documents/third_exp/train/1_9.tif")
print (img.shape)
# print (gray_image.shape)
# print (img[1])
# print (img[2])
# img = preprocess(img)
# imsave('/Users/xuchenyang/Documents/1_2.png',img)
# print (img)


