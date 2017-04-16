from keras.optimizers import Adam

import numpy as np
from PIL import Image
import os
from skimage.transform import resize
from architect import get_unet,preprocess

data_path = '/Users/xuchenyang/Documents/challange/'

image_rows = 420
image_cols = 580

def load_test_data():
    imgs_test = np.load(data_path+'imgs_test.npy')
    imgs_id = np.load(data_path+'imgs_id_test.npy')
    return imgs_test, imgs_id

path = '/Users/xuchenyang/Documents/third_exp/file/lr-5-32-50/'

def predict():
    model = get_unet()
    # model.compile(loss=dice_coef_loss, optimizer=Adam(lr=1e-5), metrics=[dice_coef])
    # model = get_unet()

    imgs_test, imgs_id = load_test_data()

    mean = np.mean(imgs_test)
    std = np.std(imgs_test)

    imgs_test = preprocess(imgs_test)

    imgs_test_source = imgs_test.astype('float32')
    imgs_test_source -= mean
    imgs_test_source /= std

    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(path+'unet.hdf5')
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_predict = model.predict(imgs_test_source, verbose=1)
    np.save(data_path+'predict.npy', imgs_mask_predict)
    print "success"

def prep(img):
    img = img.astype('float32')
    img = (img > 0.5).astype(np.uint8)  # threshold
    img = resize(img, (image_cols, image_rows), preserve_range=True)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def submission():
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load(data_path+'predict.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(data_path+file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            print (rles[i])
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')

if __name__ == '__main__':
    predict()
    submission()
    # imgs_test, imgs_id = load_test_data()
    # print (imgs_test)

