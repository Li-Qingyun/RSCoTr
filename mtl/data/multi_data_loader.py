# Modified from mmf.datasets.multi_dataset_loader
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import warnings
from typing import Dict, Iterator

from torch.utils.data.dataloader import DataLoader, Sampler

from .iteration_strategies import (IterationStrategy,
                                   RoundRobinIterationStrategy)
from .sample import Sample, SampleList, convert_batch_to_sample_list


logger = logging.getLogger(__name__)


class MultiDataLoader:
    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        iteration_strategy: IterationStrategy = None,
    ):
        if loaders is None or len(loaders) == 0:
            warnings.warn(
                "Empty loaders passed into MultiDataLoader. This can have "
                "unintended consequences."
            )

        if iteration_strategy is None:
            iteration_strategy = RoundRobinIterationStrategy(loaders)

        self._iteration_strategy = iteration_strategy
        self._loaders = loaders
        self._num_datasets = len(self.loaders)
        self.dataset_list = list(loaders.keys())
        self._iterators = {}
        self._finished_iterators = {}

        self.current_index = 0
        self.set_lengths()
        self.set_samplers()

    def set_lengths(self):
        self.lengths = {name: len(loader) for name, loader in self.loaders.items()}

    def set_samplers(self):
        self.samplers: Dict[str, Sampler] = {}
        for key, loader in self.loaders.items():
            if hasattr(loader, "sampler"):
                self.samplers[key] = loader.sampler

    def get_datasets(self):
        return [loader.dataset for loader in self.loaders.values()]

    @property
    def loaders(self) -> Dict[str, DataLoader]:
        return self._loaders

    @property
    def samplers(self) -> Dict[str, Sampler]:
        return self._samplers

    @samplers.setter
    def samplers(self, samplers: Dict[str, Sampler]):
        self._samplers = samplers

    @property
    def num_datasets(self) -> int:
        return self._num_datasets

    @property
    def iterators(self) -> Dict[str, Iterator[SampleList]]:
        return self._iterators

    @iterators.setter
    def iterators(self, iterators: Dict[str, Iterator[SampleList]]):
        self._iterators = iterators

    @property
    def current_loader(self) -> DataLoader:
        return self.loaders[self.current_dataset_name]

    @property
    def iteration_strategy(self) -> IterationStrategy:
        return self._iteration_strategy

    @property
    def current_iterator(self) -> Iterator[SampleList]:
        return self.iterators[self.current_dataset_name]

    @property
    def current_dataset_name(self) -> str:
        return self.dataset_list[self.current_index]

    @property
    def current_dataset(self) -> "torch.utils.data.Dataset, None":
        if hasattr(self.current_loader, "dataset"):
            return self.current_loader.dataset
        else:
            return None

    @property
    def first_loader(self) -> DataLoader:
        return list(self.loaders.values())[0]

    def __len__(self) -> int:
        return sum(self.lengths.values())

    def __iter__(self):
        # Clear off old iterators
        self._finished_iterators = {}
        self.iterators = {}

        for key, loader in self.loaders.items():
            self.iterators[key] = iter(loader)

        self.change_dataloader()

        return self

    def __next__(self) -> SampleList:
        """Calculation of next batch is performed using following logic.

        Current chosen iterator is set in the change_dataloader function
        based on the chosen iteration strategy which is called everytime
        prepare_batch is called.

        If we get the next batch from iterator without any StopIteration exception,
        we return it as it is. Otherwise, we have two cases:

        1. In some iteration strategies (example size proportional), each dataset
        needs to same number of epochs at any given time, we need to yield
        StopIteration exception when all iterators are finished. In turn, this
        will yield to __iter__ all reignite all of the iterators. The code will
        not reach __iter__ until unless all iterators are exhausted. An iteration
        strategy should specify this behavior through `should_exhaust_all_iterators`
        property

        2. In other cases of iteration strategies, epochs don't make sense.
        Think of a case of random (equal) proportional sampling for dataset x and y
        where x is half the size of y. When x will complete its 2 epochs, y will
        have only 1 epoch completed. **So please don't use max_epochs or epoch
        based training in this case as it won't be honored**. If an iterator is
        finished, we just reignite it in this case and finished iterators
        variable isn't used. This means that this case will never reach the
        __iter__ function ever again.


        Returns:
            SampleList: sample list instance from currently selected dataset
        """
        try:
            next_batch = next(self.current_iterator)
        except StopIteration:
            if self.iteration_strategy.should_exhaust_all_iterators:
                self._finished_iterators[self.current_dataset_name] = 1

                if len(self._finished_iterators) == self.num_datasets:
                    raise
                else:
                    self.change_dataloader()
                next_batch = next(self.current_iterator)
            else:
                iterator = iter(self.current_loader)
                self.iterators[self.current_dataset_name] = iterator
                next_batch = next(self.current_iterator)

        current_dataset_name = self.current_dataset_name
        current_task = getattr(self.current_dataset, 'task', None)

        self.change_dataloader()

        next_batch['dataset_name'] = current_dataset_name
        next_batch['task'] = current_task
        return next_batch

    def change_dataloader(self):
        choice = 0

        if self.num_datasets <= 1:
            self.current_index = choice
            return

        choice = self.iteration_strategy()

        # self._finished_iterators will always be empty in case of
        # non-proportional (equal) sampling
        while self.dataset_list[choice] in self._finished_iterators:
            choice = self.iteration_strategy()

        self.current_index = choice

    def prepare_batch(self, batch):
        if self.current_dataset and hasattr(self.current_dataset, "prepare_batch"):
            batch = self.current_dataset.prepare_batch(batch)

        self.change_dataloader()
        return batch

    def seed_sampler(self, epoch: int):
        for sampler in self.samplers.values():
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

