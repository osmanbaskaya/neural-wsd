import logging
import os
from collections import OrderedDict
from collections.abc import Iterable
from itertools import cycle
from typing import NamedTuple

import dill
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .text.dataset import WikiWordSenseDisambiguationDataset
from .text.transformers import BasicTextTransformer
from .text.transformers import PaddingTransformer
from .text.transformers import PipelineRunner
from .text.transformers import PreTrainedModelTokenize
from .text.transformers import WordpieceToTokenTransformer
from .utils import merge_params

LOGGER = logging.getLogger(__name__)


class InputFeatures(NamedTuple):
    input_ids: torch.tensor
    attention_mask: torch.tensor
    segment_ids: torch.tensor
    label_id: torch.tensor


class ProcessorFactory:
    @staticmethod
    def get_or_create(cls, cache_dir, force_create=False, *args, **kwargs):
        if cls.is_cached(cache_dir) and not force_create:
            return cls.load(cache_dir)
        else:
            LOGGER.info("Cache miss.")
            return cls(*args, **kwargs)


class WikiWordSenseDataProcessor:

    # TODO remove these params as class param.
    default_hparams = {"tokenizer": {"max_seq_len": 512}, "runner": {"batch_size": 100}}
    default_tparams = {"loader": {"shuffle": False, "num_workers": 1, "batch_size": 2}}
    cache_fn = "wiki-wsd-processor.pkl"

    def __init__(self, base_model, hparams=None, tparams=None):
        self.base_model = base_model
        self._label_encoder = None
        self.data_pipeline = None

        self._hparams = merge_params(self.default_hparams, hparams)
        self._tparams = merge_params(self.default_tparams, tparams)

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def hparams(self):
        return self._hparams

    @property
    def tparams(self):
        return self._tparams

    def _create_examples(self, transformed_data, attention_masks, labels=None):
        features = []
        if labels is None:
            labels = cycle([None])

        for input_ids, mask, label in zip(transformed_data, attention_masks, labels):
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=mask,
                    # TODO: hack - check this if it works.
                    segment_ids=[0] * len(mask),
                    label_id=label,
                )
            )

        return features

    def get_examples_and_labels(self, dataset):
        # TODO; create another class out of this and make this function abstractmethod
        pass

    def fit_transform(self, examples, labels):

        self._label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)

        self.data_pipeline = self._get_data_pipeline()
        transformed_data, context = self.data_pipeline.fit_transform(examples)
        attention_masks = context["mask"]
        return self._create_examples(transformed_data, attention_masks, labels)

    def fit(self, examples, labels):
        return self.fit_transform(examples, labels)

    def transform(self, examples, labels=None):
        if self.data_pipeline is None:
            raise ValueError("You need to fit the data first")

        if labels is not None:
            labels = self.transform_labels(labels)

        transformed_data, context = self.data_pipeline.fit_transform(examples)
        attention_masks = context["mask"]
        return self._create_examples(transformed_data, attention_masks, labels)

    def transform_labels(self, y):
        if not isinstance(y, list):
            y = [y]

        y = self._label_encoder.transform(y)
        if len(y) == 1:
            y = y[0]
        return y

    def inverse_transform_labels(self, encoded_labels):

        if not isinstance(encoded_labels, Iterable):
            encoded_labels = [encoded_labels]

        y = self._label_encoder.inverse_transform(encoded_labels)
        if len(y) == 1:
            y = y[0]
        return y

    def create_data_loader(self, data, sampler):
        if not isinstance(data, torch.utils.data.Dataset):
            raise ValueError("Should be Dataset instance")
        return DataLoader(data, sampler=sampler, **self.tparams["loader"])

    def _get_data_pipeline(self):
        # Todo: many hardcoded stuff here.
        max_seq_len = self.hparams["tokenizer"]["max_seq_len"]
        lowercase_op = BasicTextTransformer(name="text-tr", to_lowercase=True, to_ascii=True)
        tokenizer_op = PreTrainedModelTokenize(
            name="tokenizer", base_model=self.base_model, **self.hparams["tokenizer"]
        )
        wordpiece_to_token_op = WordpieceToTokenTransformer(
            name="wordpiece-to-token", base_model=self.base_model
        )
        padding_op = PaddingTransformer(name="padding-op", max_seq_len=max_seq_len)

        runner = PipelineRunner(name="runner", **self.hparams["runner"])

        # Create the pipeline
        runner | lowercase_op | tokenizer_op | wordpiece_to_token_op | padding_op

        return runner

    @classmethod
    def load(cls, cache_dir):

        d = torch.load(os.path.join(cache_dir, cls.cache_fn), pickle_module=dill)
        processor = cls(d["base_model"], d["hparams"], d["tparams"])
        processor._label_encoder = d["label_encoder"]
        processor.cache_dir = cache_dir
        processor.data_pipeline = processor._get_data_pipeline()
        return processor

    def save(self, cache_dir):
        torch.save(
            {
                "base_model": self.base_model,
                "hparams": self.hparams,
                "tparams": self.tparams,
                "label_encoder": self.label_encoder,
            },
            os.path.join(cache_dir, WikiWordSenseDataProcessor.cache_fn),
            pickle_module=dill,
        )

    @classmethod
    def is_cached(cls, cache_dir):
        return os.path.exists(os.path.join(cache_dir, cls.cache_fn))

    @staticmethod
    def create_tensor_data(features, labels_available=True):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        dataset = [all_input_ids, all_input_mask, all_segment_ids]

        # Do not add labels when no labels available (e.g., inference)
        if labels_available:
            dataset.append(torch.tensor([f.label_id for f in features], dtype=torch.long))

        return TensorDataset(*dataset)


def load_data(processor, data_dir, cache_dir, cached_data_fn, dataset_types=None):
    """This method loads preprocessed data if possible. Otherwise, it fetches all the files in
    the directory regarding with dataset_type, runs the pipeline and saves the data.
    """

    if dataset_types is None:
        # dataset_types = ["train", "dev", "test"]
        dataset_types = ["tsv", "ts"]
    datasets = OrderedDict()
    for dataset_type in dataset_types:
        cache_fn = os.path.join(cache_dir, cached_data_fn)
        if os.path.exists(cache_fn):
            LOGGER.info(f"{cache_fn} is found. Reading from it.")
            features = torch.load(cache_fn, pickle_module=dill)
        else:
            dataset = WikiWordSenseDisambiguationDataset.from_tsv(data_dir, pattern=dataset_type)
            examples = [e.text for e in dataset]
            labels = [e.label for e in dataset]
            if dataset_type == "tsv":
                # if dataset_type == "train":
                features = processor.fit_transform(examples, labels)
                processor.save(cache_dir)
            else:
                features = processor.transform(examples, labels)

            torch.save(features, cache_fn, pickle_module=dill)
        datasets[dataset_type] = processor.create_tensor_data(features)
    return datasets
