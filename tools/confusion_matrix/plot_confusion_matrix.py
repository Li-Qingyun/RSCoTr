
# -*- coding: utf-8 -*-
"""
 @Author: LQY
 @Cited: XH
 @Date: 2021/3/12 08:44
 @File: plot_confusion_matrix.py
 @Explaination:  plot_confusion_matrix
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import matplotlib.font_manager as fm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues, path="confusion_matrix.png"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    myfont = fm.FontProperties(fname='./times.ttf')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=1000)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontproperties=myfont, fontsize=20)

    cb = plt.colorbar(fraction=0.035, pad=0.05, ticks=np.linspace(0, 1, 11))
    cb.set_ticks(np.linspace(0, 1, 11))
    cb.set_ticklabels(('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
    cb.update_ticks()


    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, fontproperties=myfont)
    plt.yticks(tick_marks, classes, fontsize=10, fontproperties=myfont)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.01:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=6)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20, fontproperties=myfont)
    plt.xlabel('Predicted label', fontsize=20, fontproperties=myfont)
    plt.tick_params(bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
    plt.savefig(path, bbox_inches='tight', dpi=200)  # bbox_inches参数防止x轴标签被截断


if __name__ == '__main__':
    try:
        cm = np.load('confusion_matrix/resisc_cm_221028.npy')
        datasets = datasets.ImageFolder('data/NWPU-RESISC45/test',
                                        transforms.ToTensor())
        class_names = datasets.classes
        plot_confusion_matrix(cm, classes=class_names, normalize=True,
                              title='Normalized confusion matrix',
                              path='./confusion_matrix.tif')
    except FileNotFoundError:
        print('FileNotFound !')