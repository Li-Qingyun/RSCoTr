from typing import Dict
from copy import deepcopy

import mmcls.datasets, mmdet.datasets, mmseg.datasets
from mmcv import Config, ConfigDict

import mtl.data.iteration_strategies as strategies
from mtl.data.multi_data_loader import MultiDataLoader
from mtl.data.prepare_loader_args import prepare_dataloader_args

build_dataset = dict(
    cls=mmcls.datasets.build_dataset,
    det=mmdet.datasets.build_dataset,
    seg=mmseg.datasets.build_dataset)

build_dataloader = dict(
    cls=mmcls.datasets.build_dataloader,
    det=mmdet.datasets.build_dataloader,
    seg=mmseg.datasets.build_dataloader)

strategies_map = {
    'constant': strategies.ConstantIterationStrategy,
    'round_robin': strategies.RoundRobinIterationStrategy,  # iter base
    'random': strategies.RandomIterationStrategy,  # iter base
    'size_proportional': strategies.SizeProportionalIterationStrategy,  # epoch(may raise error)/iter base # noqa
    'repeated_sequence': strategies.RepeatedSequenceIterationStrategy,  # iter base  # noqa
    'weighted_random': strategies.WeightedRandomIterationStrategy  # iter base
}


def load_data_cfg(cfg: Config):
    data_cfg = cfg.data
    for ds in data_cfg.keys():
        _cfg = data_cfg[ds].copy()
        path = _cfg.pop('config')
        task = _cfg.pop('task')
        config = Config.fromfile(path)._cfg_dict
        config.update(_cfg)
        data_cfg[ds] = ConfigDict(dict(
            task=task, config=config))


def build_datasets(data_cfg: ConfigDict, split: str = 'train') -> dict:
    assert split in ('train', 'val', 'test')
    datasets = dict()
    for ds in data_cfg.keys():
        _cfg = data_cfg[ds].copy()
        datasets[ds] = build_dataset[_cfg.task](_cfg.config.data.get(split))
        datasets[ds].task = _cfg.task
    return datasets


def build_dataloaders(cfg: Config, distributed: bool,
                      datasets: dict, train: bool = True):
    trainval = 'train' if train else 'val'
    data_loaders = dict()
    data_cfg = cfg.data
    for ds in datasets.keys():
        task = data_cfg[ds].task
        data_loaders[ds] = build_dataloader[task](
            datasets[ds],
            **prepare_dataloader_args[trainval][task](
                distributed, data_cfg[ds].config.data,
                seed=cfg.get('seed', None),
                num_gpus=len(cfg.get('gpu_ids', [0]))))
    return data_loaders


def build_iteration_strategy(cfg: Config, data_loaders: Dict):
    _cfg = cfg.copy()
    if 'strategy' in _cfg:
        strategy_type = _cfg.strategy.pop('type')
        strategy_kwargs = _cfg.strategy
    else:
        strategy_type = 'round_robin'
        strategy_kwargs = dict()
    IterationStrategy = strategies_map[strategy_type]
    iteration_strategy = IterationStrategy(data_loaders, **strategy_kwargs)
    print(f"{'#'*30}\nIteration Strategy has been prepared.")
    i_s = IterationStrategy(data_loaders, **strategy_kwargs)
    print(i_s.config.name)
    get_iter_list = lambda iter_s: [iter_s() for _ in range(300)]
    _get_p_list = lambda iter_list: [iter_list.count(i) for i in range(len(data_loaders))]
    get_p_list = lambda iter_list: [val / min(_get_p_list(iter_list))
                                    for val in _get_p_list(iter_list)]
    iter_list = get_iter_list(i_s)
    print(f'{iter_list}\n{get_p_list(iter_list)}\n{"#"*30}')
    return iteration_strategy


def build_multidataloader(cfg: Config, distributed: bool, datasets: dict):
    # build train loaders
    train_loaders = build_dataloaders(cfg, distributed, datasets)

    # build iteration strategy
    iteration_strategy = build_iteration_strategy(cfg, train_loaders)

    # build multidataloader
    multi_dataloader = MultiDataLoader(train_loaders, iteration_strategy)
    return multi_dataloader


if __name__ == '__main__':
    data = dict(
        dior=dict(
            task='det',
            config='configs/_base_/det/dior.py',
            data=dict(
                samples_per_gpu=100)),
        potsdam=dict(
            task='seg',
            config='configs/_base_/seg/potsdam_IRRG_all.py',
            data=dict(
                samples_per_gpu=100)))
    cfg = dict(
        data=data,
        gpu_ids='0',
        seed=2022)
    cfg = Config(cfg)
    load_data_cfg(cfg)
    # train_loader (MultiDataLoader)
    datasets = build_datasets(cfg.data)
    data_loader = build_multidataloader(cfg, False, datasets)
    iter(data_loader)
    x0 = next(data_loader)
    x1 = next(data_loader)
    x2 = next(data_loader)
    x3 = next(data_loader)
    # val_loaders (Dict[str, Dataloaders])
    valsets = build_datasets(cfg.data, split='val')
    val_loaders = build_dataloaders(cfg, False, valsets, train=False)
    # test_loaders (Dict[str, Dataloaders])
    testsets = build_datasets(cfg.data, split='test')
    test_loaders = build_dataloaders(cfg, False, testsets, train=False)
