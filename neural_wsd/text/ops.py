import multiprocessing
from copy import deepcopy

import torch
from itertools import cycle
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import AutoTokenizer


class Operator:

    inputs = []
    outputs = []

    def __init__(self, op_name="", next_op=None):
        self.op_name = op_name
        self.next_op = next_op

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

    def fit(self, data):
        self._validate_requirements(data)

        if not self.is_fitted:
            self._fit(data)
            self._is_fitted = True

        return self.transform(data)

    def transform(self, data):
        self._validate_requirements(data)
        if not self.is_fitted:
            raise ValueError("Need to fit this first.")

        if self.pool is not None:
            it = cycle(Operator._batch_iter(data, self.batch_size))
            transformed = self.pool.map(self._transform, it)
        else:
            transformed = self._transform(data)

        return transformed

    def fit_transform(self, data):
        return self.fit(data)

    def _fit(self, data):
        raise NotImplementedError()

    def _transform(self, data):
        raise NotImplementedError()

    def _validate_requirements(self, data):
        if not all(key in data for key in self.inputs):
            # TODO more info for the error
            raise ValueError("Pipeline is not feasiable.")

    @staticmethod
    def _batch_iter(data, batch_size):
        for index in range(0, len(data), batch_size):
            yield data[index : index + batch_size]

    def set_pool(self, pool):
        self._pool = pool

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def __or__(self, other):
        self.next_op = other
        return other

    def __repr__(self):
        return f"{self.op_name} - {super().__repr__()}"


class StatelessOperator(Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_fitted = True

    def _fit(self, data):
        pass

    def _transform(self, data):
        raise NotImplementedError()


class PreTrainedModelTokenizeOp(StatelessOperator):
    def __init__(
        self, base_model=None, max_length=512, do_lower_case=True, do_basic_tokenize=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model, do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize
        )

    def _transform(self, data):
        data["default"] = [
            self._tokenizer.encode(sample, add_special_tokens=True) for sample in data["default"]
        ]

        return data


class BasicTextProcessingOp(StatelessOperator):
    # TODO: Add more text preprocessing operations.
    def __init__(self, lowercase=True, **kwargs):
        super().__init__(**kwargs)
        self.lowercase = lowercase

    def _transform(self, data):
        transformed = data["default"]
        if self.lowercase:
            transformed = [sample.lower() for sample in transformed]

        data["default"] = transformed
        return data


class PipelineRunner(Operator):
    def __init__(self, num_process=1, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.num_process = num_process
        if self.num_process > 1:
            self.set_pool(multiprocessing.Pool())

        self.set_batch_size(batch_size)
        self.pipeline = None

    @staticmethod
    def get_pipeline(initial_op):
        op = initial_op.next_op
        pipeline = []
        while op is not None:
            pipeline.append(op)
            op = op.next_op
        return pipeline

    def _fit(self, data):
        self.pipeline = PipelineRunner.get_pipeline(self)
        if self.num_process > 1:
            for op in self.pipeline:
                op.set_pool(self.pool)
                op.set_batch_size(self.batch_size)

        transformed = deepcopy(data)
        if not isinstance(transformed, dict):
            transformed = {"default": transformed}
        for op in self.pipeline:
            transformed = op.fit(transformed)
        return transformed

    def _transform(self, data):
        transformed = deepcopy(data)
        if not isinstance(transformed, dict):
            transformed = {"default": transformed}
        for op in self.pipeline:
            transformed = op.transform(transformed)
        return transformed

    def validate_pipeline(self):
        pass


class PaddingOp(StatelessOperator):
    def __init__(self, max_length=512, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def _transform(self, data):
        # TODO check pad_seq params.

        transformed = data["default"]
        transformed = pad_sequences(
            transformed, self.max_length, padding="post", truncating="post", value=0
        )

        mask = torch.tensor(transformed != 0, dtype=torch.int64, requires_grad=False)

        data["default"] = transformed
        data["mask"] = mask

        return data
