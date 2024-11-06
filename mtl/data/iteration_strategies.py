# Modified from mmf.datasets.iteration_strategies
# Copyright (c) Facebook, Inc. and its affiliates.


from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader


class IterationStrategy:
    """
    Base class for defining iteration strategies that will be used
    for iterating over multiple datasets during multitasking.

    An IterationStrategy implementation should `__call__` method
    which returns index of dataset from which next batch must be
    pulled.

    Class can also define `should_exhaust_all_iterators` property
    which defines whether all iterators should be exhausted before
    reigniting next batch of iterators. For example, in size
    proportional iteration strategy, all iterators must be finished
    before starting a new round so that all of them get equal
    opportunity to present themselves according to their size.

    Args:
        config (Config): Object of type Config which should be defined
            for each iteration strategy for configurable parameters.
        dataloaders (Dict[str, DataLoader]): A dictionary containing
            mapping from dataset key to its dataloader.

    Usage::

        from dataclasses import dataclass
        from mmf.datasets.iterators import IterationStrategy

        class MyStrategy(IterationStrategy):
            @dataclass
            class Config:
                name: str = "my_strategy"
            def __init__(self, config, dataloader):
                ...
    """

    @dataclass
    class Config:
        name: str = MISSING

    def __init__(
        self, dataloaders: Dict[str, DataLoader], config: Config = None, *args, **kwargs
    ):
        if config is None:
            config = {}
        config = OmegaConf.merge(OmegaConf.structured(self.Config), config)
        self.config = config
        self.dataloaders = dataloaders

    @classmethod
    def from_params(cls, dataloaders: Dict[str, DataLoader], **kwargs):
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(dataloaders, config)

    @property
    def should_exhaust_all_iterators(self) -> bool:
        return False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ hasn't been implemented")


class ConstantIterationStrategy(IterationStrategy):
    """
    Always returns a constant number. Useful for mimicing single task
    training in multitask setup for verification or defaults purposes

    index to be returned can be specified in config parameter as `idx`.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "constant"
        idx: int = 0

    def __init__(
        self, dataloaders: Dict[str, DataLoader], config: Config = None, *args, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)
        self._idx = self.config.idx

    @property
    def should_exhaust_all_iterators(self) -> bool:
        return True

    def __call__(self, *args, **kwargs):
        return self._idx


class RoundRobinIterationStrategy(IterationStrategy):
    """
    Samples datasets one by one in round robin fashion.

    Start index can be specified in config as `start_idx`.

    Also defaults to size proportional sampling as roundrobin
    doesn't make sense with validation and testing splits
    as they need to finish one complete epoch.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "round_robin"
        start_idx: int = 0

    def __init__(
        self, dataloaders: Dict[str, DataLoader], config: Config = None, *args, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)

        if "start_idx" in self.config:
            self._current_idx = self.config.start_idx

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.dataloaders)
        return nxt


class RepeatedSequenceIterationStrategy(IterationStrategy):

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "repeated_sequence"

    def __init__(
        self, dataloaders: Dict[str, DataLoader], sequence, *args, config: Config = None, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)
        self._current_idx = 0
        assert max(sequence) == len(dataloaders) - 1 and \
               min(sequence) == 0 and \
               len(np.unique(sequence)) == len(dataloaders)
        self.sequence = sequence

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.sequence)
        choice = self.sequence[nxt]
        return choice


class RandomIterationStrategy(IterationStrategy):
    """
    Samples random number each time when sampled.

    Follows test/validation strategy similar to RoundRobin.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "random"

    def __init__(
        self, dataloaders: Dict[str, DataLoader], config: Config = None, *args, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(len(self.dataloaders), 1)[0]  # TODO: use torch
        # TODO: There may be some question, the output chance is actually not uniform.  # noqa
        return choice


class WeightedRandomIterationStrategy(IterationStrategy):
    """
    Samples random number each time when sampled.

    Follows test/validation strategy similar to RoundRobin.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "weighted_random"

    def __init__(
        self, dataloaders: Dict[str, DataLoader], p, *args, config: Config = None, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)
        assert len(p) == len(dataloaders)
        p_sum = sum(p)
        if p_sum != 1:
            print('p_sum != 1, and it have been converted to 1.')
            self.p = [val / p_sum for val in p]
        assert sum(self.p) == 1

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(len(self.dataloaders), 1, p=self.p)[0]
        return choice


class SizeProportionalIterationStrategy(IterationStrategy):
    """
    Samples index based on size of each dataset. Bigger datasets
    are sampled more and this strategy requires completing
    all iterators before starting new ones.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "size_proportional"

    def __init__(
        self, dataloaders: Dict[str, DataLoader], config: Config = None, *args, **kwargs
    ):
        super().__init__(dataloaders, config, *args, **kwargs)
        self._per_dataset_lengths = []
        self._total_length = 0

        for loader in self.dataloaders.values():
            # Some loaders might not have dataset attribute
            # set, in this case we need to fail gracefully as we can't
            # calculate lengths.
            assert hasattr(loader, "dataset"), (
                "loaders need dataset objects to work with "
                + "'size_proportional' sampling"
            )

            dataset_instance = loader.dataset

            len_msg = "all datasets should have __len__ defined " \
                      "to work with proportional sampling iterator"
            _len = getattr(dataset_instance, '__len__', lambda: None)
            assert callable(_len), len_msg
            dataset_instance_length = _len()
            assert dataset_instance_length is not None, len_msg
            assert (
                dataset_instance_length
            ), f"dataset: {dataset_instance} is empty"
            self._per_dataset_lengths.append(dataset_instance_length)
            self._total_length += dataset_instance_length

        self._dataset_probabilities = self._per_dataset_lengths[:]
        self._dataset_probabilities = [
            prob / self._total_length for prob in self._dataset_probabilities
        ]

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataset_probabilities
        )[0]  # TODO: use torch
        return choice

    @property
    def should_exhaust_all_iterators(self):
        return True

