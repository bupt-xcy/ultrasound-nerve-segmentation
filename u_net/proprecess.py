import glob
import os

import cv2
import numpy as np
import shutil
import skimage
import utils
from skimage.util import view_as_blocks
from skimage.io import imsave, imread
import scipy.spatial.distance as spdist


TRAIN_PATH = '/Users/xuchenyang/Downloads/KaggleNerveSeg/train/'
smooth = 1

def dice_coef(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def load_and_preprocess(imgname):
    img_fname = imgname
    mask_fname = os.path.splitext(imgname)[0] + "_mask.tif"
    img = cv2.imread(os.path.join(TRAIN_PATH, img_fname), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    mask = cv2.imread(os.path.join(TRAIN_PATH, mask_fname), cv2.IMREAD_GRAYSCALE)
    assert mask is not None

    # newsize = (img.shape[1] / 4, img.shape[0] / 4)
    # newsize = (128, 128)
    # img = cv2.resize(img, newsize)
    # mask = cv2.resize(mask, newsize)
    mask = (mask > 128).astype(np.float32)

    # TODO: Could subtract mean as on fimg above
    img = img.astype(np.float32) / 255.0
    np.ascontiguousarray(img)
    return img, mask


def load_patient(pid):
    fnames = [os.path.basename(fname) for fname in glob.glob(TRAIN_PATH + "/%d_*.tif" % pid) if 'mask' not in fname]
    imgs, masks = zip(*map(load_and_preprocess, fnames))
    imgs = np.array(imgs)
    masks = np.array(masks)
    return imgs, masks, fnames



def compute_img_hist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)


OUTDIR = "/Users/xuchenyang/Documents/clean/"
if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.mkdir(OUTDIR)


def filter_images_for_patient(pid):
    imgs, masks, fnames = load_patient(pid)
    hists = np.array(map(compute_img_hist, imgs))
    D = spdist.squareform(spdist.pdist(hists, metric='cosine'))

    # Used 0.005 to train at 0.67
    close_pairs = D + np.eye(D.shape[0]) < 0.008

    close_ij = np.transpose(np.nonzero(close_pairs))

    incoherent_ij = [(i, j) for i, j in close_ij if dice_coef(masks[i], masks[j]) < 0.2]
    incoherent_ij = np.array(incoherent_ij)

    # i, j = incoherent_ij[np.random.randint(incoherent_ij.shape[0])]

    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in incoherent_ij:
        if np.sum(masks[i]) == 0:
            valids[i] = False
        if np.sum(masks[j]) == 0:
            valids[i] = False

    for i in np.flatnonzero(valids):
        imgname = os.path.splitext(fnames[i])[0] + ".tif"
        mask_fname = os.path.splitext(imgname)[0] + "_mask.tif"
        print(imgs[i].shape)
        img = skimage.img_as_ubyte(imgs[i])
        print(imgs[i].shape)
        cv2.imwrite(os.path.join(OUTDIR, imgname), img)
        print(masks[i].shape)
        mask = skimage.img_as_ubyte(masks[i])
        print(imgs[i].shape)
        cv2.imwrite(os.path.join(OUTDIR, mask_fname), mask)
    print 'Discarded ', np.count_nonzero(~valids), " images for patient %d" % pid


for pid in range(1, 47):
    filter_images_for_patient(pid)

print('success')
