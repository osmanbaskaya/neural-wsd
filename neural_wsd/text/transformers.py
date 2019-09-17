import multiprocessing
from copy import deepcopy
from itertools import cycle
from typing import NamedTuple

import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder


# ---------------- Feature Transformers -------------------


class BaseTransformer:
    inputs = []
    outputs = []

    def __init__(self, name="", next_transformer=None):
        self.name = name
        self.next_transformer = next_transformer

        self._is_fitted = False
        self._pool = None
        self._batch_size = None

    @property
    def pool(self):
        return self._pool

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_fitted(self):
        return self._is_fitted

    def fit(self, data, context):
        self._validate_requirements(data)

        if not self.is_fitted:
            self._fit(data, context)
            self._is_fitted = True

        return self

    def transform(self, data, context=None):
        if context is None:
            context = {}
        self._validate_requirements(data)
        if not self.is_fitted:
            raise ValueError("Need to fit this first.")

        if self.pool is not None:
            it = cycle(BaseTransformer._batch_iter(data, self.batch_size))
            # TODO make sure that all data is unfolded.
            transformed = self.pool.map(self._transform, it)
        else:
            transformed = self._transform(data, context)

        return transformed

    def fit_transform(self, data, context=None):
        return self.fit(data, context).transform(data, context)

    def _fit(self, data, context):
        raise NotImplementedError()

    def _transform(self, data, context):
        raise NotImplementedError()

    def _validate_requirements(self, data):
        if not all(key in data for key in self.inputs):
            # TODO more info for the error
            raise ValueError("Pipeline is not feasible.")

    @staticmethod
    def _batch_iter(data, batch_size):
        for index in range(0, len(data), batch_size):
            yield data[index : index + batch_size]

    def set_pool(self, pool):
        self._pool = pool

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def __or__(self, other):
        self.next_transformer = other
        return other

    def __repr__(self):
        return f"{self.name} - {super().__repr__()}"


class StatelessBaseTransformer(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_fitted = True

    def _fit(self, data, context):
        return self

    def _transform(self, data, context):
        raise NotImplementedError()


class PreTrainedModelTokenize(StatelessBaseTransformer):
    def __init__(
        self,
        base_model=None,
        max_seq_len=512,
        do_lower_case=True,
        do_basic_tokenize=False,
        num_of_special_tokens=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.max_length = max_seq_len
        self.num_of_special_tokens = num_of_special_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model, do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize
        )

    def _transform(self, data, context):
        data = [
            self._tokenizer.encode(
                sample[: self.max_length - self.num_of_special_tokens], add_special_tokens=True
            )
            for sample in data
        ]

        return data, context


class BasicTextTransformer(StatelessBaseTransformer):
    # TODO: Add more text preprocessing operations.
    def __init__(self, lowercase=True, **kwargs):
        super().__init__(**kwargs)
        self.lowercase = lowercase

    def _transform(self, data, context):
        transformed = data
        if self.lowercase:
            transformed = [sample.lower() for sample in transformed]

        return transformed, context


class PipelineRunner(BaseTransformer):
    def __init__(self, num_process=1, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.num_process = num_process
        if self.num_process > 1:
            self.set_pool(multiprocessing.Pool())

        self.set_batch_size(batch_size)
        self.pipeline = None

    def create_pipeline(self):
        t = self.next_transformer
        pipeline = []
        while t is not None:
            pipeline.append(t)
            t = t.next_transformer
        self.pipeline = pipeline

    def _fit(self, data, context):
        self.create_pipeline()
        if self.num_process > 1:
            for transformer in self.pipeline:
                transformer.set_pool(self.pool)
                transformer.set_batch_size(self.batch_size)

        transformed = deepcopy(data)
        for transformer in self.pipeline:
            transformed, context = transformer.fit_transform(transformed, context)

    def _transform(self, data, context=None):
        if self.pipeline is None:
            self.create_pipeline()
        if context is None:
            context = {}
        transformed = deepcopy(data)
        for transformer in self.pipeline:
            transformed, context = transformer.transform(transformed, context)
        return transformed, context

    def validate_pipeline(self):
        pass


class PaddingTransformer(StatelessBaseTransformer):
    def __init__(self, max_seq_len=512, padding="post", truncating="post", value=0, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.truncating = truncating
        self.value = value
        self.max_length = max_seq_len

    def _transform(self, data, context):
        # TODO check pad_seq params.

        transformed = pad_sequences(
            data,
            self.max_length,
            padding=self.padding,
            truncating=self.truncating,
            value=self.value,
        )

        # todo dtype check this
        mask = (transformed != 0).astype("int64")

        if "mask" in context:
            raise ValueError("context has the key `mask`")

        context["mask"] = mask
        return transformed, context
