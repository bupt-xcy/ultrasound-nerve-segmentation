from skimage.io import imsave, imread
import numpy as np
import os


def isAllBlack():
    data_path = "/Users/xuchenyang/Documents/third_exp/"
    train_data_path = os.path.join(data_path, 'validation')
    images = os.listdir(train_data_path)
    print (len(images)/2)
    black = 0
    for image_name in images:
        if 'mask' in image_name:
            img = imread(os.path.join(train_data_path, image_name), as_grey=True)
            a = np.array(img)
            if(a.flatten().sum()==0):
                black=black+1
    return black

print (isAllBlack())