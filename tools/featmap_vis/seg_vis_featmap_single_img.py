import torch
from mmcv import Config, imread
from mmseg.models import build_backbone
from cls_vis_featmap_single_img import get_one_img_one_lvl_one_channel_featmap


if __name__ == '__main__':
    CFG_PATH = 'configs/seg/RSCoTrSeg_3scale_swin-t-p4-w7_512x512_80k_potsdam_IRRG_all.py'
    CKPT_PATH = 'work_dirs/seg_single_w_mtl_wo_task_pretrain/MTL_slvlcls_swin-t-p4-w7_1x1_potsdam/best_potsdam_aAcc_iter_54800.pth'  # seg_single

    config = Config.fromfile(CFG_PATH)
    backbone_cfg = config.model.backbone
    backbone = build_backbone(backbone_cfg)
    backbone.eval()

    ckpt = torch.load(CKPT_PATH)
    state_dict = ckpt['state_dict']
    state_dict = {n[9:]: p for n, p in state_dict.items() if n.startswith('backbone.')}
    print(backbone.load_state_dict(state_dict, strict=False))

    IMG_PATH = r'data/potsdam/img_IRRG/train/2_10_0_0_512_512.png'

    img_name = '2_10_0_0_512_512'
    image = imread(IMG_PATH, channel_order='rgb')

    LVL, CHANNEL = 0, 1
    folder_name = 'seg_single'
    specific_map = get_one_img_one_lvl_one_channel_featmap(
        backbone, image, LVL, CHANNEL,
        with_img=False,
        path_prefix=f'./{folder_name}/{IMG_PATH.split("/")[-1][:-4]}')
    specific_map = get_one_img_one_lvl_one_channel_featmap(
        backbone, image, LVL, CHANNEL,
        path_prefix=f'./{folder_name}/{IMG_PATH.split("/")[-1][:-4]}')
