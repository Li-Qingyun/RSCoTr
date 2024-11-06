# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import numpy as np
import matplotlib.font_manager as fm

import mmcv
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmcv import Config, DictAction
from mmcv.image import imresize
from mmcv.cnn import MODELS
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.core.visualization import imshow_det_bboxes

from mmdet.utils import (build_ddp, build_dp, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from mtl.data.build import build_datasets, build_dataloaders, load_data_cfg
from mtl.data.multi_eval_dataset import MultiEvalDatasets
from mtl.engine import single_gpu_test, multi_gpu_test

from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--tasks', nargs='+', type=str,
        choices=['cls', 'det', 'seg'], default=['cls', 'det', 'seg'])
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test_outputs', type=str, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_cls_results(cls_out, class_names):
    if isinstance(cls_out, list):
        cls_out = cls_out[0]
    if isinstance(cls_out, dict):
        assert 'pred_class' in cls_out
        cls_out = cls_out['pred_class']
    print(class_names[np.argmax(cls_out)])


def draw_det_results(img, det_results, class_names, out_file=None):
    img = imresize(img, (800, 800))
    bboxes = np.vstack(det_results[0])
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(det_results[0])
    ]
    labels = np.concatenate(labels)
    img = imshow_det_bboxes(img, bboxes, labels, class_names=class_names,
                            out_file=out_file, score_thr=0.3)
    return img


def draw_seg_results(img, seg_results, class_names, palette, out_file=None, opacity=0.5):
    seg = seg_results[0]
    palette = np.array(palette)
    assert palette.shape[0] == len(class_names)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img


if __name__ == '__main__':
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # load configs from dataset config files
    load_data_cfg(cfg)

    # build the dataloader
    datasets = build_datasets(cfg.data, split='test')
    datasets = {name: MultiEvalDatasets(dataset)
                for name, dataset in datasets.items()
                if dataset.task in args.tasks}
    CLASSES = {name: dataset.CLASSES for name, dataset in datasets.items()}
    dataloaders = build_dataloaders(cfg, distributed, datasets, train=False)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the model and load checkpoint
    cfg.model.train_cfg = {name: {} for name in cfg.model.train_cfg.keys()}
    model = MODELS.build(cfg.model)
    model.eval()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = {
            name: dataset.CLASSES for name, dataset in datasets.items()}

    # IMG_FILEPATH = 'data/potsdam/img_IRRG/train/2_10_0_0_512_512.png'
    # IMG_FILEPATH = 'data/potsdam/img_IRRG/train/2_10_0_512_512_1024.png'
    IMG_FILEPATH = 'data/potsdam/img_IRRG/train/2_10_512_4608_1024_5120.png'

    FILEPATH = IMG_FILEPATH.replace('img_IRRG', 'img_RGB')
    img_irrg = mmcv.imread(IMG_FILEPATH)
    img = mmcv.imread(FILEPATH)
    pipeline_inputs = dict(
        img_info=dict(filename=FILEPATH.split('/')[-1]),
        ori_filename=FILEPATH.split('/')[-1],
        filename=FILEPATH,
        img=img,
        img_irrg=img_irrg,
        img_shape=img.shape,
        ori_shape=img.shape,
        img_fields=['img'],
        img_prefix=None)


    def false_collate_fn(pipeline_out):
        if isinstance(pipeline_out['img'], list):
            pipeline_out['img'] = [img.unsqueeze(0)
                                   for img in pipeline_out['img']]
        else:
            pipeline_out['img'] = [pipeline_out['img'].unsqueeze(0)]
        if isinstance(pipeline_out['img_metas'], list):
            pipeline_out['img_metas'] = [meta.data
                                         for meta in pipeline_out['img_metas']]
        else:
            pipeline_out['img_metas'] = [pipeline_out['img_metas'].data]
        return pipeline_out


    cls_pipeline_inputs = deepcopy(pipeline_inputs)
    cls_pipeline = datasets['resisc'].dataset.pipeline
    cls_pipeline.transforms.pop(0)
    cls_input_data = false_collate_fn(cls_pipeline(cls_pipeline_inputs))
    cls_out = model.forward_test('cls', **cls_input_data)
    get_cls_results(cls_out, class_names=model.CLASSES['resisc'])

    seg_pipeline_inputs = deepcopy(pipeline_inputs)
    seg_pipeline_inputs['img'] = seg_pipeline_inputs['img_irrg']
    seg_pipeline = datasets['potsdam'].dataset.pipeline
    seg_palette = datasets['potsdam'].dataset.PALETTE
    seg_pipeline.transforms.pop(0)
    seg_input_data = false_collate_fn(seg_pipeline(seg_pipeline_inputs))
    seg_out = model.forward_test('seg', **seg_input_data)
    draw_seg_results(img, seg_out, class_names=model.CLASSES['potsdam'],
                     palette=seg_palette, out_file='./seg_output.png')

    det_pipeline_inputs = deepcopy(pipeline_inputs)
    det_pipeline = datasets['dior'].dataset.pipeline
    det_pipeline.transforms.pop(0)
    det_input_data = false_collate_fn(det_pipeline(det_pipeline_inputs))
    det_out = model.forward_test('det', **det_input_data)
    draw_det_results(img, det_out, class_names=model.CLASSES['dior'],
                     out_file='./det_output.png')
