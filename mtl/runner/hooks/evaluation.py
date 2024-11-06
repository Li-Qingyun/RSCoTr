from typing import Optional, Callable, List, Dict, Sequence

from torch.utils.data import DataLoader

from mmcv.utils import is_seq_of
from mmcv.runner.hooks import EvalHook


class KeyIndicator:

    def __init__(self, **kwargs):
        self.key_indicator = dict(**kwargs)

    def __getitem__(self, item):
        return self.key_indicator.__getitem__(item)

    def __repr__(self):
        keys = list(self.key_indicator.keys())
        keys = [key.replace('.', '_') for key in keys]
        return '_'.join(keys)

    def __len__(self):
        return len(self.key_indicator)

    def items(self):
        return self.key_indicator.items()


class MultiDatasetsEvalHook(EvalHook):

    def __init__(self,
                 dataloaders: Dict[str, DataLoader],
                 start: Optional[int] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 save_best: Optional[str] = None,
                 # rule: Optional[str] = None,
                 test_fn: Optional[Callable] = None,
                 greater_keys: Optional[List[str]] = None,
                 less_keys: Optional[List[str]] = None,
                 out_dir: Optional[str] = None,
                 file_client_args: Optional[dict] = None,
                 **eval_kwargs):
        super(EvalHook, self).__init__()
        if not isinstance(dataloaders, dict):
            raise TypeError(f'dataloaders must be a dict of pytorch DataLoader'
                            f', but got {type(dataloaders)}')
        for loader in dataloaders.values():
            if not isinstance(loader, DataLoader):
                raise TypeError(f'dataloader must be a pytorch DataLoader, '
                                f'but got {type(loader)}')

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloaders = dataloaders
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, (str, list, dict)) or save_best is None, \
            '""save_best"" should be a str, or list, or dict, or None ' \
            f'rather than {type(save_best)}'
        if isinstance(save_best, str):
            save_best = {save_best: 1}
        if isinstance(save_best, list):
            save_best = {key: 1 for key in save_best}
        self.save_best = save_best

        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        if test_fn is None:
            from mtl.engine import single_gpu_test
            self.test_fn = single_gpu_test
        else:
            self.test_fn = test_fn

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                assert isinstance(greater_keys, str)
                greater_keys = (greater_keys, )
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                assert isinstance(greater_keys, str)
                less_keys = (less_keys, )
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule('greater', self.save_best)

        self.out_dir = out_dir
        self.file_client_args = file_client_args

    def _init_rule(self, rule: Optional[str], key_indicator: dict):
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')
        self.rule = rule
        self.key_indicator = KeyIndicator(**key_indicator)
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def _do_evaluate(self, runner):
        results_dict = self.test_fn(runner.model, self.dataloaders)
        runner.log_buffer.output['eval_iter_num'] = {
            name: len(dataloader) for name, dataloader in self.dataloaders.items()}
        key_score = self.evaluate(runner, results_dict)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results_dict):
        eval_res = dict()
        for dataset_name, dataloader in self.dataloaders.items():
            task = getattr(dataloader.dataset, 'task')
            eval_res.update(
                {f'{dataset_name}.{metric_name}': val
                 for metric_name, val in dataloader.dataset.evaluate(
                    results_dict[dataset_name], logger=runner.logger,
                    **self.eval_kwargs.get(task, None)).items()})

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            metrics_sum = sum([eval_res.get(key, 0.) * weight
                               for key, weight in self.key_indicator.items()])
            metrics_sum /= len(self.key_indicator)
            return metrics_sum

        return None