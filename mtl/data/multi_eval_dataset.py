from copy import deepcopy

from torch.utils.data import Dataset

from mmcv.parallel import DataContainer


class MultiEvalDatasets(Dataset):
    def __init__(self, dataset: Dataset):
        assert hasattr(dataset, 'task')
        self.dataset = dataset
        self.task = getattr(dataset, 'task')

    def __getitem__(self, item):
        data = self.dataset.__getitem__(item)
        img_metas = deepcopy(data['img_metas'])
        if isinstance(img_metas, list):
            img_metas = img_metas[0]
        data['task'] = DataContainer(
            self.task,
            stack=img_metas.stack,
            padding_value=img_metas.padding_value,
            cpu_only=img_metas.cpu_only,
            pad_dims=img_metas.pad_dims)
        return data

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'task: {self.task} ' + repr(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)