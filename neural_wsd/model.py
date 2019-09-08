from pytorch_transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
)
import pandas as pd
import os
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")


# ---------------- Feature Transformers -------------------


class Tokenize:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        copied = deepcopy(sample)
        # TODO (obaskaya): replace encoding with tokenize & convert to address truncation problem.
        copied["text"] = self.tokenizer.encode(copied["text"].lower(), add_special_tokens=True)
        copied["length"] = len(copied["text"])
        return copied


class Padding:
    def __init__(self, max_length=512):
        self.max_length = max_length

    def __call__(self, sample):
        copied = deepcopy(sample)
        padded = pad_sequences(
            [sample["text"]], self.max_length, padding="post", truncating="post", value=0
        )

        copied["text"] = torch.tensor(padded[0], dtype=torch.int64)

        mask = torch.zeros(max_length, requires_grad=False, dtype=torch.int64)
        mask[: copied["length"]] = 1  # mark only the sentence tokens, not padding tokens.
        copied["attention_mask"] = mask

        return copied


class WikiWordSenseDisambiguationDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transforms):
        self.directory = directory
        self.__label_encoder = LabelEncoder()
        self.transforms = transforms
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

        for func in self.transforms:
            sample = func(sample)
        return sample

    @property
    def num_of_unique_labels(self):
        return len(self.__label_encoder.classes_)


class PreTrainedNeuralDisambiguator:
    def __init__(self, num_labels, base_model="bert-base-uncased"):
        self.config = BertConfig.from_pretrained(base_model, num_labels=num_labels)
        self.model = BertForSequenceClassification.from_pretrained(base_model, config=self.config)

    def predict(self, text):
        # preprocess
        # return torch.argmax(self.model(text))
        pass


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, do_basic_tokenize=False
)

max_length = 512
tokenize = Tokenize(tokenizer=tokenizer, max_length=max_length)
pad = Padding(max_length=max_length)
dataset = WikiWordSenseDisambiguationDataset("dataset", transforms=[tokenize, pad])
data_loader = DataLoader(dataset, 5, shuffle=False, num_workers=1)
batches = list(data_loader)
model = PreTrainedNeuralDisambiguator(dataset.num_of_unique_labels)
print(model)
