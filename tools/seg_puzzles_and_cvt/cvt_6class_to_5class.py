import os

import numpy as np
from mmcv import imread, imwrite

# SEG_MAP_FOLDER_PATH = r'/media/lqy/Elements SE/show_results/RSCoTrSeg_output_potsdam/show_1_whole'
SEG_MAP_FOLDER_PATH = r'/media/lqy/Elements SE/RS_MTL_vis/seg_vis/seg_results_vis/show_seg_whole'
SAVE_PATH = SEG_MAP_FOLDER_PATH + '_cvt-out'


if __name__ == '__main__':
    for dir_entry in os.scandir(SEG_MAP_FOLDER_PATH):
        name, path = dir_entry.name[:-4], dir_entry.path
        img = imread(path)  # bgr
        red_filter = np.all(img == (0, 0, 255), axis=2)
        out = img + np.array([255, 255, 0], dtype=img.dtype)[None, None] * red_filter[:, :, None]
        out_path = os.path.join(SAVE_PATH, f'{name}_cvt.png')
        imwrite(out, out_path)
        print(f'{out_path} has been saved !')
