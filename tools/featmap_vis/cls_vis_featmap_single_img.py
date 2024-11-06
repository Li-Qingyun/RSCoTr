from pathlib import Path
import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as T

import mmcv
from mmcv import Config, imread
from mmdet.models import build_backbone

from draw_featmap import draw_featmap


def preprocess_image(img: np.ndarray, mean: list, std: list) -> Tensor:
    img = np.float32(img) / 255
    preprocessing = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def select_sub_fig(whole_fig, index):
    h, w, _ = whole_fig.shape
    num_h, num_w = int(h / 256), int(w / 256)
    assert index < num_h * num_w
    index_0 = index // num_w
    index_1 = index % num_w
    return whole_fig[index_0 * 256: (index_0 + 1) * 256,
                     index_1 * 256: (index_1 + 1) * 256]


def get_one_img_featmap(model, image, with_img=True, with_text=True, topk=20):
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    feat = model(input_tensor)
    out_feat_maps = []
    for lvl in range(4):
        lvl_feat_map_list = []
        drawn_img = draw_featmap(feat[lvl][0], image if with_img else None,
                                 channel_reduction=None, topk=topk,
                                 arrangement=(4, 5),
                                 resize_shape=(256, 256),
                                 with_text=with_text)
        lvl_feat_map_list.append(drawn_img)
        for k in range(topk):
            lvl_feat_map_list.append(select_sub_fig(drawn_img, k))
        out_feat_maps.append(lvl_feat_map_list)
    return out_feat_maps


def get_one_img_one_lvl_one_channel_featmap(
        model, image, lvl, channel,
        with_img=True, with_text=True, path_prefix=None):
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    feat = model(input_tensor)
    drawn_img = draw_featmap(feat[lvl][0], image if with_img else None,
                             channel_reduction=None,
                             channel_id=channel,
                             resize_shape=(256, 256),
                             with_text=with_text)
    if path_prefix is not None:
        save_path = path_prefix + f'_level_{lvl}_channel_{channel}{"" if with_img else "_without_img"}.png'
        mmcv.imwrite(drawn_img[:, :, ::-1], str(save_path), auto_mkdir=True)
    return drawn_img


def save_featmaps(featmaps, folder_path):
    folder_path = Path(folder_path)

    for lvl, lvl_featmaps in enumerate(featmaps):
        for k, featmap in enumerate(lvl_featmaps):
            out_path = folder_path / f'level_{lvl}_num_{k}.png'
            mmcv.imwrite(featmap[:, :, ::-1], str(out_path), auto_mkdir=True)


if __name__ == '__main__':
    CFG_PATH = 'configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py'
    # CKPT_PATH = 'work_dirs/round_robin_1-1-0.1/best_resisc_accuracy_top-1_dior_bbox_mAP_potsdam_mFscore_iter_300000.pth'  # MTL
    # CKPT_PATH = 'work_dirs/cls_single_w_mtl_wo_task_pretrain/MTL_slvlcls_swin-t-p4-w7_1x1_resisc/best_resisc_accuracy_top-1_iter_80000.pth'  # cls_single
    # CKPT_PATH = '/home/lqy/Desktop/pretrained/dino_swin-t-p4-w7_mmdet.pth'  # det_single
    CKPT_PATH = 'work_dirs/seg_single_w_mtl_wo_task_pretrain/MTL_slvlcls_swin-t-p4-w7_1x1_potsdam/best_potsdam_aAcc_iter_54800.pth'  # seg_single

    config = Config.fromfile(CFG_PATH)
    backbone_cfg = config.model.backbone
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    ckpt = torch.load(CKPT_PATH)
    state_dict = ckpt['state_dict']
    state_dict = {n[9:]: p for n, p in state_dict.items() if n.startswith('backbone.')}
    print(backbone.load_state_dict(state_dict, strict=False))

    # IMG_PATH = r'data/NWPU-RESISC45/test/airplane/airplane_001.jpg'
    # IMG_PATH = r'data/DIOR/JPEGImages-test/11726.jpg'
    IMG_PATH = r'data/potsdam/img_IRRG/train/2_10_0_0_512_512.png'

    # img_name = 'airplane_001'
    # img_name = '11726'
    img_name = '2_10_0_0_512_512'
    # SAVE_FOLDER_PATH = f'{img_name}_feat_maps'
    # SAVE_FOLDER_PATH = f'{img_name}_feat_maps_without_img'
    # SAVE_FOLDER_PATH = f'{img_name}_feat_maps_without_channel_index'
    # SAVE_FOLDER_PATH = f'{img_name}_feat_maps_without_img_without_channel_index'

    image = imread(IMG_PATH, channel_order='rgb')

    # save_featmaps(get_one_img_featmap(backbone, image), SAVE_FOLDER_PATH)
    # save_featmaps(get_one_img_featmap(backbone, image, with_img=False), SAVE_FOLDER_PATH)
    # save_featmaps(get_one_img_featmap(backbone, image, with_text=False), SAVE_FOLDER_PATH)
    # save_featmaps(get_one_img_featmap(backbone, image, with_img=False, with_text=False), SAVE_FOLDER_PATH)

    LVL, CHANNEL = 3, 18
    # folder_name = 'cls_single'
    # folder_name = 'det_single'
    folder_name = 'seg_single'
    specific_map = get_one_img_one_lvl_one_channel_featmap(
        backbone, image, LVL, CHANNEL,
        with_img=False,
        path_prefix=f'./{folder_name}/{IMG_PATH.split("/")[-1][:-4]}')
    specific_map = get_one_img_one_lvl_one_channel_featmap(
        backbone, image, LVL, CHANNEL,
        path_prefix=f'./{folder_name}/{IMG_PATH.split("/")[-1][:-4]}')
