import numpy as np
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    GT_LABELS_PATH = r'confusion_matrix/resisc_gt_labels_221028.npy'
    PRED_PATH = r'confusion_matrix/resisc_mtl_pred_221028.npy'

    gt_labels = np.load(GT_LABELS_PATH)
    pred = np.load(PRED_PATH)
    pred_labels = np.argmax(pred, axis=1)

    cm = confusion_matrix(gt_labels, pred_labels)
    np.save('confusion_matrix/resisc_cm_221028.npy', cm)
