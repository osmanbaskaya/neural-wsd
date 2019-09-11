from abc import abstractmethod

import torch
from pytorch_transformers import BertForSequenceClassification, BertConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_wsd.text.dataset import WikiWordSenseDisambiguationDataset
from neural_wsd.text.transformers import (
    PreTrainedModelTokenize,
    BasicTextTransformer,
    PipelineRunner,
    PaddingTransformer,
)


class BaseModel:
    _hparams = {}
    _tparams = {}

    def __init__(self):
        self._label_encoder = None
        self.data_pipeline = None
        self.model = None
        self.data_loader = None

    @property
    def hparams(self):
        return self._hparams

    @property
    def tparams(self):
        return self._tparams

    def fit_to_labels(self, data):
        self._label_encoder = LabelEncoder().fit(data)

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

    def train(self, data):
        self.data_pipeline = self._get_data_pipeline()
        self.model = self._get_model()

        self.fit_to_labels(data.labels)

        train_dl, validation_dl = self.create_data_loader(data)

        # train_epoch_it = tqdm(train_dl, desc="Training Iteration")
        # validation_epoch_it = tqdm(validation_dl, desc="Validation Iteration")
        for step, batch in enumerate(list(train_dl)):
            input_ids, context = self.data_pipeline.fit_transform(batch["text"])
            attention_mask = context["mask"]
            labels = self.transform_labels(batch["label"])
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            labels = torch.tensor(labels)
            loss, logits = self.model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            print(loss, logits)

    def create_data_loader(self, data):
        if isinstance(data, torch.utils.data.Dataset):
            train_dl = DataLoader(data, **self.tparams["loader"])
        else:
            pass  # TODO: Implement later.

        # TODO implement here later.
        return train_dl, []

    def override_hparams(self, params):
        BaseModel._override_params(self.hparams, params)

    def override_tparams(self, params):
        BaseModel._override_params(self.tparams, params)

    @staticmethod
    def _override_params(params_to_change, params):
        # TODO: use toolz to override.
        pass

    @abstractmethod
    def _get_data_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        X, context = self.data_pipeline.fit_transform(texts)
        attention_mask = context["mask"]

        logits = self.model(
            input_ids=torch.tensor(X, dtype=torch.int64), attention_mask=attention_mask
        )

        return torch.argmax(logits)


class PreTrainedNeuralDisambiguator(BaseModel):
    _hparams = {"tokenizer": {"max_length": 512}, "model": {}, "runner": {"batch_size": 100}}

    _tparams = {"loader": {"shuffle": False, "num_workers": 1, "batch_size": 2}}

    def __init__(self, base_model="bert-base-uncased", num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

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

    def _get_model(self):
        bert_config = BertConfig.from_pretrained(self.base_model, num_labels=self.num_labels)
        return BertForSequenceClassification.from_pretrained(self.base_model, config=bert_config)


dataset = WikiWordSenseDisambiguationDataset(directory="dataset")
model = PreTrainedNeuralDisambiguator(
    base_model="bert-base-uncased", num_labels=dataset.num_of_unique_labels
)
model.train(dataset)
model.predict(["here is some data", "some more"])
print(model)
