from __future__ import print_function

import os
import numpy as np
from PIL import Image
from skimage.io import imread
import cv2

data_path = "/Users/xuchenyang/Downloads/"

image_rows = 420
image_cols = 580


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path+'imgs_test.npy', imgs)
    np.save(data_path+'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')

if __name__ == '__main__':
    create_test_data()
