from cls_vis_featmap import *


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