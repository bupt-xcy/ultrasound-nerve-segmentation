import numpy as np
from predict import load_test_data
from train import preprocess


path = '/Users/xuchenyang/Documents/third_exp/file/segnet-lr-3-32-100/'
predicted_masks = np.load( path + 'predict.npy')

imgs_test, imgs_test_mask = load_test_data()
#imgs_test_source = imgs_test.astype('float32')
imgs_test_gt = preprocess(imgs_test_mask)

predicted_masks_flat = predicted_masks.flatten()
test_gt_masks_flat = imgs_test_gt.flatten()

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(test_gt_masks_flat, predicted_masks_flat, pos_label=255)

import matplotlib.pyplot as plt
#plt.plot(list(fpr),list(tpr))
plt.plot([0,1],[0,1],'k--')
line1, = plt.plot(fpr,tpr,'b',label="U-NET ROC (AUC = 0.86)")

plt.legend(handles=[line1],loc=4,prop={'size':12})
#plt.plot(list(fpr),list(tpr))
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.grid()
plt.savefig(path+'roc')
# plt.show()

from sklearn.metrics import roc_auc_score
test_gt_masks_flat = test_gt_masks_flat/255
auc = roc_auc_score(test_gt_masks_flat,predicted_masks_flat)
auc = np.array([auc])
np.savetxt(path+'auc.txt', auc)
print (auc)