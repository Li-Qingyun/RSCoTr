from pathlib import Path
import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import mmcv
from mmcv import Config, imread
from mmdet.models import build_backbone

from draw_featmap import draw_featmap


class ImageFolderWrapper(ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = imread(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


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


def get_one_img_featmap(model, image, topk=20):
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    feat = model(input_tensor)
    out_feat_maps = []
    for lvl in range(4):
        lvl_feat_map_list = []
        drawn_img = draw_featmap(feat[lvl][0], image,
                                 channel_reduction=None, topk=topk,
                                 arrangement=(4, 5),
                                 resize_shape=(256, 256))
        lvl_feat_map_list.append(drawn_img)
        for k in range(topk):
            lvl_feat_map_list.append(select_sub_fig(drawn_img, k))
        out_feat_maps.append(lvl_feat_map_list)
    return out_feat_maps


def save_featmaps(featmaps, ori_img_path):
    ori_img_path = Path(ori_img_path)
    assert ori_img_path.parts[-3] == 'resisc_img'
    folder_path = '/'.join([*ori_img_path.parts[:-3],
                            'resisc_feat',
                            ori_img_path.parts[-2],
                            ori_img_path.parts[-1][:-4]])
    folder_path = Path(folder_path)

    for lvl, lvl_featmaps in enumerate(featmaps):
        for k, featmap in enumerate(lvl_featmaps):
            out_path = folder_path / f'level_{lvl}_num_{k}.png'
            mmcv.imwrite(featmap[:, :, ::-1], str(out_path), auto_mkdir=True)


if __name__ == '__main__':
    config = Config.fromfile('configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py')
    backbone_cfg = config.model.backbone
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    ckpt = torch.load('work_dirs/round_robin_1-1-0.1/best_resisc_accuracy_top-1_dior_bbox_mAP_potsdam_mFscore_iter_300000.pth')
    state_dict = ckpt['state_dict']
    state_dict = {n[9:]: p for n, p in state_dict.items() if n.startswith('backbone.')}
    print(backbone.load_state_dict(state_dict))

    RESISC_IMG_PATH = '/media/lqy/Elements SE/vis/resisc_img'
    dataset = ImageFolderWrapper(RESISC_IMG_PATH)

    for index in range(len(dataset)):
        img, _, img_path = dataset[index]
        save_featmaps(get_one_img_featmap(backbone, img), img_path)