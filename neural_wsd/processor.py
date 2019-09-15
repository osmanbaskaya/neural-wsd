import logging
import toolz
import dill
import os
from collections import OrderedDict
from typing import NamedTuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from .text.dataset import WikiWordSenseDisambiguationDataset
from .text.transformers import (
    PreTrainedModelTokenize,
    BasicTextTransformer,
    PipelineRunner,
    PaddingTransformer,
)

LOGGER = logging.getLogger(__name__)


class InputFeatures(NamedTuple):
    input_ids: torch.tensor
    input_mask: torch.tensor
    segment_ids: torch.tensor
    label_id: torch.tensor


class ProcessorFactory:
    @staticmethod
    def get_or_create(cls, cache_dir, *args, **kwargs):
        if cls.is_cached(cache_dir):
            return cls.load(cache_dir)
        else:
            LOGGER.info("Cache miss.")
            return cls(*args, **kwargs)


class WikiWordSenseDataProcessor:

    default_hparams = {"tokenizer": {"max_length": 512}, "runner": {"batch_size": 100}}
    default_tparams = {"loader": {"shuffle": False, "num_workers": 1, "batch_size": 2}}
    cache_fn = "wiki-wsd-processor.pkl"

    def __init__(self, base_model, hparams=None, tparams=None):
        self.base_model = base_model
        self._label_encoder = None
        self.data_pipeline = None

        if hparams is None:
            hparams = {}
        if tparams is None:
            tparams = {}

        self._hparams = toolz.merge(WikiWordSenseDataProcessor.default_hparams, hparams)

        self._tparams = toolz.merge(WikiWordSenseDataProcessor.default_tparams, tparams)

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def hparams(self):
        return self._hparams

    @property
    def tparams(self):
        return self._tparams

    def _create_examples(self, transformed_data, attention_masks, dataset):
        features = []
        for i in range(len(transformed_data)):
            features.append(
                InputFeatures(
                    input_ids=transformed_data[i],
                    input_mask=attention_masks[i],
                    # TODO: hack - check this if it works.
                    segment_ids=[0] * len(attention_masks[i]),
                    label_id=float(self.transform_labels([dataset[i].label])),
                )
            )
        return features

    def fit_transform(self, dataset):

        self._label_encoder = LabelEncoder()
        self.label_encoder.fit(dataset.labels)

        self.data_pipeline = self._get_data_pipeline()
        texts = [e.text for e in dataset]
        transformed_data, context = self.data_pipeline.fit_transform(texts)
        attention_masks = context["mask"]
        return self._create_examples(transformed_data, attention_masks, dataset)

    def fit(self, dataset):
        return self.fit_transform(dataset)

    def transform(self, dataset):
        if self.data_pipeline is None:
            raise ValueError("You need to fit the data first")

        transformed_data, context = self.data_pipeline.fit_transform([e.text for e in dataset])
        attention_masks = context["mask"]
        return self._create_examples(transformed_data, attention_masks, dataset)

    def transform_labels(self, y):
        try:
            return self._label_encoder.transform(y)
        except ValueError:
            return self._label_encoder.transform([y])[0]

    def inverse_transform_labels(self, encoded_labels):
        try:
            return self._label_encoder.inverse_transform(encoded_labels)
        except ValueError:
            return self._label_encoder.inverse_transform([encoded_labels])[0]

    def create_data_loader(self, data, sampler):
        if not isinstance(data, torch.utils.data.Dataset):
            raise ValueError("Should be Dataset instance")
        return DataLoader(data, sampler=sampler, **self.tparams["loader"])

    def _get_data_pipeline(self):
        # Todo: many hardcoded stuff here.
        lowercase_op = BasicTextTransformer(name="text-tr", lowercase=True)
        tokenizer_op = PreTrainedModelTokenize(
            name="tokenizer", base_model=self.base_model, **self.hparams["tokenizer"]
        )
        padding_op = PaddingTransformer(name="padding-op")

        runner = PipelineRunner(name="runner", **self.hparams["runner"])

        # Create the pipeline
        runner | lowercase_op | tokenizer_op | padding_op

        return runner

    @classmethod
    def load(cls, cache_dir):

        d = torch.load(os.path.join(cache_dir, cls.cache_fn), pickle_module=dill)
        processor = cls(d["base_model"], d["hparams"], d["tparams"])
        processor._label_encoder = d["label_encoder"]
        processor.cache_dir = cache_dir
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
    def create_tensor_data(features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


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
            if dataset_type == "tsv":
                # if dataset_type == "train":
                features = processor.fit_transform(dataset)
                processor.save(cache_dir)
            else:
                features = processor.transform(dataset)

            torch.save(features, cache_fn, pickle_module=dill)
        datasets[dataset_type] = processor.create_tensor_data(features)
    return datasets
