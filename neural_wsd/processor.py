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

from .text.dataset import FeatureDataset
from .text.dataset import WikiWordSenseDisambiguationDataset
from .text.transformers import BasicTextTransformer
from .text.transformers import PaddingTransformer
from .text.transformers import PipelineRunner
from .text.transformers import PreTrainedModelTokenize
from .text.transformers import WordPieceListTransformer
from .utils import merge_params

LOGGER = logging.getLogger(__name__)


class InputFeatures(NamedTuple):
    """Encapsulate all features. If any additional feature is used for any new processor, just
    add the feature's name, type, and default value."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor = None
    alignment: list = None
    offset: torch.Tensor = 0  # offset of a target word in question of disambiguation.
    label: torch.Tensor = None


class ProcessorFactory:
    @staticmethod
    def get_or_create(cls, cache_dir, ignore_cache=False, *args, **kwargs):
        if cls.is_cached(cache_dir) and not ignore_cache:
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

    @staticmethod
    def create_examples(feature_dict):
        """General InputFeatures creation. It can handle features with/without alignments/offset
        info."""

        features = []
        keys = feature_dict.keys()
        flatten_values = zip(*feature_dict.values())
        for v in flatten_values:
            features.append(InputFeatures(**dict(zip(keys, v))))

        return features

    def process_dataset(self, dataset, *, fit_first=True):
        transformed, context = self._process_dataset(dataset, fit_first)
        return self.__class__.create_examples(transformed)

    def _process_dataset(self, dataset, fit_first=True):

        examples = [e.text for e in dataset]
        labels = [e.label for e in dataset]

        if fit_first:
            transformed, context = self.fit_transform(examples, labels)
        else:
            transformed, context = self.transform(examples, labels)

        return transformed, context

    def fit_transform(self, examples, labels):
        self._label_encoder = LabelEncoder().fit(labels)
        self.data_pipeline = self._get_data_pipeline()
        return self.transform(examples, labels)

    def fit(self, examples, labels):
        return self.fit_transform(examples, labels)

    def transform(self, examples, labels=None):
        """Basic transformation. It returns only basic elements. Subclasses can add more stuff
        such as offset, or alignment information.
        """
        transformed_data, context, labels = self._transform(examples, labels)
        d = {"input_ids": transformed_data, "attention_mask": context["mask"]}

        if labels is not None:
            d["label"] = labels

        return d, context

    def _transform(self, examples, labels):
        """Basic transformation."""
        if self.data_pipeline is None:
            raise ValueError("You need to fit the data first")

        if labels is not None:
            labels = self.transform_labels(labels)

        transformed_data, context = self.data_pipeline.fit_transform(examples)
        return transformed_data, context, labels

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
        wordpiece_to_token_op = WordPieceListTransformer(
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
    def _create_tensor_features(features):
        if len(features) == 0:
            raise ValueError("Features should contain more than 1 element.")

        field_types = InputFeatures._field_types

        dataset = []
        data_order = []

        for key, value in field_types.items():
            if issubclass(value, torch.Tensor):
                # Check that the `key` field of InputFeature is defined.
                # We have to do this check since this method is general. Some InputFeatures may be
                # None (e.g., alignment, offset). We just # want to skip for those that are None.
                if features[0].__getattribute__(key) is not None:
                    data = [feature.__getattribute__(key) for feature in features]
                    dataset.append(torch.tensor(data, dtype=torch.long))
                    data_order.append(key)

        return TensorDataset(*dataset), data_order

    def create_tensor_data(self, features: InputFeatures):
        # TODO: Create a more general version of this. Iterate over all the features and
        # if there is an attribute, and it's tensor form, then put it into database.
        # https://stackoverflow.com/questions/6570075/getting-name-of-value-from-namedtuple
        # as_dict may work. If it works, remove `labels_available` attribute.
        # If necessary, refactor InputFeatures.

        tensor_features, order = self.__class__._create_tensor_features(features)
        return FeatureDataset(tensor_features, input_order=order)


class WikiTokenBaseProcessor(WikiWordSenseDataProcessor):
    def _process_dataset(self, dataset, fit_first=True):
        transformed, context = super()._process_dataset(dataset, fit_first)

        # Add offsets to default processing.
        offsets = []
        miss = longest = 0
        for i, alignment in enumerate(transformed["alignment"]):
            offset = dataset[i].offset
            # check target word is inside of the alignment boundaries.
            if offset > longest:
                longest = offset
            if offset < len(alignment):
                offsets.append(offset)
            else:
                offsets.append(0)  # use the first token to disambiguate.
                miss += 1

        LOGGER.info(
            f"For {(miss / len(offsets) * 100):.4}% target words, first token will be used to "
            f"disambiguate the sense, since max_length is too short to cover those instances."
        )
        LOGGER.info(f"Biggest offset is {longest}")
        transformed["offset"] = offsets
        return transformed, context

    def transform(self, examples, labels=None):
        d, context = super().transform(examples, labels)

        # TODO: Is it necessary to check if this key is in the context? It should be, no?
        if "wordpiece_to_token_list" in context:
            d["alignment"] = context["wordpiece_to_token_list"]

        return d, context

    def create_tensor_data(self, features: InputFeatures):
        tensor_features, order = self.__class__._create_tensor_features(features)
        return FeatureDataset(
            tensor_features=tensor_features,
            input_order=order,
            alignments=[f.alignment for f in features],
        )


def load_data(processor, data_dir, cache_dir, dataset_types=None, ignore_cache=False):
    """This method loads preprocessed data if possible. Otherwise, it fetches all the files in
    the directory regarding with dataset_type, runs the pipeline and saves the data.
    """

    if dataset_types is None:
        # dataset_types = ["train", "test"]
        dataset_types = ["tsv", "ts"]
    datasets = OrderedDict()
    for dataset_type in dataset_types:
        if os.path.exists(cache_dir) and not ignore_cache:
            LOGGER.info(f"{cache_dir} is found. Reading from it.")
            feature_dataset = FeatureDataset.load(cache_dir)
        else:
            dataset = WikiWordSenseDisambiguationDataset.from_tsv(data_dir, pattern=dataset_type)
            if dataset_type == "tsv":
                # if dataset_type == "train":
                features = processor.process_dataset(dataset, fit_first=True)
                processor.save(cache_dir)
            else:
                # This is test/validation data, should not fit.
                features = processor.process_dataset(dataset, fit_first=False)
            feature_dataset = processor.create_tensor_data(features)
            feature_dataset.save(cache_dir)
        datasets[dataset_type] = feature_dataset
    return datasets
