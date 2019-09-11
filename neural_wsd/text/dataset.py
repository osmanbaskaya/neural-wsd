import os

import pandas as pd
import torch.utils.data


class WikiWordSenseDisambiguationDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.__data = WikiWordSenseDisambiguationDataset.read_data_to_dataframe(directory)
        self._labels = self.data["sense"].unique()
        self._num_of_unique_labels = len(self._labels)

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
        sample = {
            "text": row["tokenized_sentence"],
            "label": row["sense"],
            "offset": row["offset"],
            "word": row["target_word"],
            "id": row["id"],
        }

        return sample
