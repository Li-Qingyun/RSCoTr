from mmcv import Config

import mmcls.datasets, mmdet.datasets, mmseg.datasets
from mmdet.utils.compat_config import compat_loader_args
from mmdet.datasets import replace_ImageToTensor


def prepare_cls_trainloader_args(distributed, data_cfg, num_gpus, seed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        round_up=True,
        seed=seed)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **data_cfg.get('train_dataloader', {})}
    return train_loader_cfg


def prepare_cls_valloader_args(distributed, data_cfg, num_gpus, seed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        round_up=True,
        seed=seed)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    val_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **data_cfg.get('val_dataloader', {}),
    }
    return val_loader_cfg


def prepare_cls_testloader_args(distributed, data_cfg, num_gpus, seed):
    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        round_up=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']})
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **data_cfg.get('test_dataloader', {})}
    return test_loader_cfg


def prepare_det_trainloader_args(distributed, data_cfg, num_gpus, seed):
    _cfg = Config(dict(data=data_cfg))
    _cfg = compat_loader_args(_cfg)
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        seed=seed,
        # runner_type='EpochBasedRunner',  # TODO: Iter. makes batch_size 1
        runner_type='IterBasedRunner',  # TODO: Iter. makes batch_size 1
        persistent_workers=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **_cfg.data.get('train_dataloader', {})}
    return train_loader_cfg


def prepare_det_valloader_args(distributed, data_cfg, num_gpus, seed):
    _cfg = Config(dict(data=data_cfg))
    _cfg = compat_loader_args(_cfg)
    val_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False,
        persistent_workers=False)
    val_dataloader_args = {
        **val_dataloader_default_args,
        **_cfg.data.get('val_dataloader', {})}
    if val_dataloader_args['samples_per_gpu'] > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        _cfg.data.val.pipeline = mmdet.datasets.replace_ImageToTensor(
            _cfg.data.val.pipeline)
    return val_dataloader_args


def prepare_det_testloader_args(distributed, data_cfg, num_gpus, seed):
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(data_cfg.test, dict):
        data_cfg.test.test_mode = True
        if data_cfg.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            data_cfg.test.pipeline = replace_ImageToTensor(
                data_cfg.test.pipeline)
    elif isinstance(data_cfg.test, list):
        for ds_cfg in data_cfg.test:
            ds_cfg.test_mode = True
        if data_cfg.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in data_cfg.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **data_cfg.get('test_dataloader', {})}
    return test_loader_cfg


def prepare_seg_trainloader_args(distributed, data_cfg, num_gpus, seed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        seed=seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']})
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **data_cfg.get('train_dataloader', {})}
    return train_loader_cfg


def prepare_seg_valloader_args(distributed, data_cfg, num_gpus, seed):
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        seed=seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']})
    # The specific dataloader settings
    val_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **data_cfg.get('val_dataloader', {})}
    return val_loader_cfg


def prepare_seg_testloader_args(distributed, data_cfg, num_gpus, seed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=num_gpus,
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in data_cfg.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']})
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **data_cfg.get('test_dataloader', {})}
    return test_loader_cfg


prepare_dataloader_args = dict(
    train=dict(
        cls=prepare_cls_trainloader_args,
        det=prepare_det_trainloader_args,
        seg=prepare_seg_trainloader_args),
    val=dict(
        cls=prepare_cls_valloader_args,
        det=prepare_det_valloader_args,
        seg=prepare_seg_valloader_args),
    test=dict(
        cls=prepare_cls_testloader_args,
        det=prepare_det_testloader_args,
        seg=prepare_seg_testloader_args))