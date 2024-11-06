import os
from collections import defaultdict

import numpy as np
from mmcv import imread, imwrite

# PIECES_FOLDER_PATH = r'./show_seg_0.5'
PIECES_FOLDER_PATH = r'./show_seg'
SAVE_PATH = PIECES_FOLDER_PATH + '_whole'


def collect_puzzles_info(pieces_folder_path):
    puzzles_info = defaultdict(list)
    convert_pos = lambda xs: tuple(int(x) for x in xs)
    for dir_entry in os.scandir(pieces_folder_path):
        name, path = dir_entry.name, dir_entry.path
        name_parts = name[:-4].split('_')
        whole_fig_name = f'{name_parts[0]}_{name_parts[1]}'
        puzzles_info[whole_fig_name].append(
            dict(path=path, pos=convert_pos(name_parts[2:])))
    return puzzles_info


def get_whole_fig(pieces_list):
    max_pos = np.array([piece['pos'] for piece in pieces_list]).max(axis=0)
    w_whole, h_whole = max_pos[2], max_pos[3]
    whole_fig = 255 * np.ones([h_whole, w_whole, 3], dtype=np.uint8)
    for piece in pieces_list:
        piece_img = imread(piece['path'], channel_order='rgb')
        left, up, right, down = piece['pos']
        whole_fig[up:down, left:right] = piece_img
    return whole_fig


if __name__ == '__main__':
    puzzles_info = collect_puzzles_info(PIECES_FOLDER_PATH)
    for whole_fig_name, pieces_list in puzzles_info.items():
        whole_fig_path = os.path.join(SAVE_PATH, whole_fig_name + '.png')
        whole_fig = get_whole_fig(pieces_list)
        imwrite(whole_fig[:, :, ::-1], whole_fig_path, auto_mkdir=True)
        print(f'{whole_fig_path} has been saved!')
