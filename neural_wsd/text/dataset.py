import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class WikiWordSenseDisambiguationDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.__label_encoder = LabelEncoder()
        self.__data = WikiWordSenseDisambiguationDataset.read_data_to_dataframe(directory)
        self.__fit_to_labels()

    @property
    def data(self):
        return self.__data

    def __fit_to_labels(self):
        self.__label_encoder.fit(self.data["sense"].unique())

    def transform_labels(self, y):
        try:
            return self.__label_encoder.transform(y)
        except ValueError:
            return self.__label_encoder.transform([y])[0]

    def inverse_transform_labels(self, encoded_labels):
        try:
            return self.__label_encoder.inverse_transform(encoded_labels)
        except ValueError:
            return self.__label_encoder.inverse_transform([encoded_labels])[0]

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
            "label": self.transform_labels(row["sense"]),
            "offset": row["offset"],
            "word": row["target_word"],
            "id": row["id"],
        }

        return sample

    @property
    def num_of_unique_labels(self):
        return len(self.__label_encoder.classes_)
