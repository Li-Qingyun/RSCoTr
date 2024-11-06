import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as T

import mmcv
from mmcv import Config, imread
from mmdet.models import build_backbone

from mmengine.visualization import Visualizer
visualizer = Visualizer()


def preprocess_image(img: np.ndarray, mean: list, std: list) -> Tensor:
    img = np.float32(img) / 255
    preprocessing = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def select_sub_fig(whole_fig, index):
    h, w, _ = drawn_img.shape
    num_h, num_w = int(h / 256), int(w / 256)
    assert index < num_h * num_w
    index_0 = index // num_w
    index_1 = index % num_w
    return whole_fig[index_0 * 256: (index_0 + 1) * 256,
                     index_1 * 256: (index_1 + 1) * 256]


if __name__ == '__main__':
    config = Config.fromfile('configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py')
    backbone_cfg = config.model.backbone
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    ckpt = torch.load('work_dirs/round_robin_1-1-0.1/best_resisc_accuracy_top-1_dior_bbox_mAP_potsdam_mFscore_iter_300000.pth')
    state_dict = ckpt['state_dict']
    state_dict = {n[9:]: p for n, p in state_dict.items() if n.startswith('backbone.')}
    print(backbone.load_state_dict(state_dict))

    image = imread('data/NWPU-RESISC45/test/airplane/airplane_001.jpg', channel_order='rgb')

    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    feat = backbone(input_tensor)


    # drawn_img = visualizer.draw_featmap(feat[-1][0],
    #                                     # channel_reduction='select_max',
    #                                     topk=20, arrangement=(4, 5)
    #                                     )
    drawn_img = visualizer.draw_featmap(feat[-1][0], image,
                                        # channel_reduction='select_max',
                                        # channel_reduction='squeeze_mean',
                                        channel_reduction=None, topk=20, arrangement=(4, 5), resize_shape=(256, 256)
                                        )
    # visualizer.show(drawn_img)
    mmcv.imshow(drawn_img[:, :, ::-1])
    mmcv.imshow(select_sub_fig(drawn_img, 5)[:, :, ::-1])