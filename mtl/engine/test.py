from typing import Dict, List, Optional

from torch import nn
from torch.utils.data import DataLoader

import mmcv.engine
import mmcls.apis
import mmdet.apis
import mmseg.apis

single_gpu_single_dataset_test = dict(
    cv=mmcv.engine.single_gpu_test,
    cls=mmcls.apis.single_gpu_test,
    det=mmdet.apis.single_gpu_test,
    seg=mmseg.apis.single_gpu_test)

multi_gpu_single_dataset_test = dict(
    cv=mmcv.engine.multi_gpu_test,
    cls=mmcls.apis.multi_gpu_test,
    det=mmdet.apis.multi_gpu_test,
    seg=mmseg.apis.multi_gpu_test)


def single_gpu_test(model: nn.Module,
                    data_loaders: Dict[str, DataLoader],
                    show: bool = False,
                    out_dir: str = None,
                    kwargs_dict: Dict = None) -> Dict[str, List]:
    results = dict()
    CLASSES = model.CLASSES
    kwargs_dict = dict() if kwargs_dict is None else kwargs_dict
    for name, dataloader in data_loaders.items():
        task = getattr(dataloader.dataset, 'task', 'cv')
        kwargs = kwargs_dict.get(task, dict())
        model.CLASSES = CLASSES[name]
        results[name] = single_gpu_single_dataset_test[task](
            model, dataloader, show, out_dir, **kwargs)
    return results


def multi_gpu_test(model: nn.Module,
                   data_loaders: Dict[str, DataLoader],
                   tmpdir: Optional[str] = None,
                   gpu_collect: bool = False,
                   kwargs_dict: Dict = None) -> Dict[str, Optional[list]]:
    results = dict()
    kwargs_dict = dict() if kwargs_dict is None else kwargs_dict
    for name, dataloader in data_loaders.items():
        task = getattr(dataloader.dataset, 'task', 'cv')
        kwargs = kwargs_dict.get(task, dict())
        results[name] = multi_gpu_single_dataset_test[task](
            model, dataloader, tmpdir, gpu_collect, **kwargs)
    return results
