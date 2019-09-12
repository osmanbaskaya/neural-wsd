import os
import random

import math
import pandas as pd
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

SEED = 42


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


class WikiWordSenseDisambiguationDataset(torch.utils.data.Dataset):
    def __init__(self, data=None, label_column=None):
        if label_column is None:
            label_column = "sense"
        self.__data = data
        self._labels = self.data[label_column].unique()
        self._num_of_unique_labels = len(self._labels)

    @staticmethod
    def from_tsv(directory, label_column=None):
        data = WikiWordSenseDisambiguationDataset.read_data_to_dataframe(directory)
        return WikiWordSenseDisambiguationDataset(data=data, label_column=label_column)

    @property
    def data(self):
        return self.__data

    @property
    def num_of_unique_labels(self):
        return self._num_of_unique_labels

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def read_data_to_dataframe(directory, column_names=None):
        # TODO: replace here with a general solution
        if column_names is None:
            column_names = [
                "id",
                "target_word",
                "offset",
                "sense",
                "annotated_sentence",
                "tokenized_sentence",
                "sentence",
            ]
        all_data = []
        for filename in os.listdir(directory)[:2]:
            fn = os.path.join(directory, filename)
            print(fn)
            all_data.append(
                pd.read_csv(fn, delimiter="\t", names=column_names, header=None, index_col=False)
            )

        return pd.concat(all_data, sort=False)

    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Todo: replace here with dict(**row).
        sample = {
            "text": row["tokenized_sentence"],
            "label": row["sense"],
            "offset": row["offset"],
            "word": row["target_word"],
            "id": row["id"],
        }

        return sample
