# coding=utf-8
import csv
import glob
import logging
import math
import pathlib
import random
from abc import abstractmethod
from typing import NamedTuple

import dill
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

SEED = 42
LOGGER = logging.getLogger("__name___")


def sample_data(dataset, size_of_first_dataset=0.8, shuffle=True, seed=SEED):
    """Split the `dataset` into two and returns a sampler for each split"""

    random.seed(seed)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    first_dataset_size = int(math.ceil(size_of_first_dataset * dataset_size))

    if shuffle:
        random.shuffle(indices)

    first_data_indices = indices[:first_dataset_size]
    second_data_indices = indices[first_dataset_size:]
    return SubsetRandomSampler(first_data_indices), SubsetRandomSampler(second_data_indices)


class WordSenseDisambiguationDataset(torch.utils.data.Dataset):
    def __init__(self, data=None):
        self._examples, labels = self._create_examples(data)
        self._labels = list(labels)
        self._num_labels = len(self._labels)

    @classmethod
    def from_tsv(cls, directory, pattern):
        files = glob.glob(f"{directory}/*{pattern}*", recursive=True)
        LOGGER.info(f"{len(files)} will be read.")
        if len(files) == 0:
            raise ValueError("Check the directory/pattern. no file found.")
        all_lines = cls._read_all_tsv_files(files)
        return cls(data=all_lines)

    @property
    def examples(self):
        return self._examples

    @property
    def num_of_labels(self):
        return self._num_labels

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def _read_all_tsv_files(files):
        lines = []
        for fn in files:
            LOGGER.info(fn)
            with open(fn, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                lines.extend([line for line in reader])
        return lines

    @abstractmethod
    def _create_examples(self, data):
        raise NotImplementedError

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class WikiWordSenseDisambiguationDataset(WordSenseDisambiguationDataset):
    """Basic Wikipedia Word Sense Disambiguation dataset."""

    class Example(NamedTuple):
        id: int
        target_word: str
        offset: int
        label: str
        text: str

    def _create_examples(self, data):
        examples = []
        labels = set()
        for d in data:
            e = WikiWordSenseDisambiguationDataset.Example(
                id=d[0],
                target_word=d[1],
                offset=int(d[2]),
                label=d[3],
                text=d[5],  # taking tokenized text.
            )
            examples.append(e)
            labels.add(e.label)

        return examples, labels


class FeatureDataset:

    _tensor_fn = "tensor.pkl"
    _input_order_fn = "input_order.pkl"
    _alignment_fn = "alignment.pkl"

    def __init__(self, tensor_features, input_order, alignments=None):
        self._tensor_features = tensor_features
        self._input_order = input_order
        self._alignments = alignments

    def save(self, cache_dir):
        cache_dir = pathlib.Path(cache_dir)
        torch.save(self._tensor_features, cache_dir / self.__class__._tensor_fn, pickle_module=dill)
        torch.save(
            self._input_order, cache_dir / self.__class__._input_order_fn, pickle_module=dill
        )
        if self._alignments is not None:
            torch.save(
                self._alignments, cache_dir / self.__class__._alignment_fn, pickle_module=dill
            )

    @classmethod
    def load(cls, cache_dir):
        cache_dir = pathlib.Path(cache_dir)

        tensor_dataset = torch.load(cache_dir / cls._tensor_fn, pickle_module=dill)
        input_order = torch.load(cache_dir / cls._input_order_fn, pickle_module=dill)

        alignments = None
        alignment_fn = cache_dir / cls._alignment_fn
        if alignment_fn.exists():
            alignments = torch.load(alignment_fn, pickle_module=dill)

        return cls(tensor_dataset, input_order, alignments)

    @property
    def tensor_dataset(self):
        return self._tensor_features

    @property
    def alignments(self):
        return self._alignments

    @property
    def input_order(self):
        return self._input_order
