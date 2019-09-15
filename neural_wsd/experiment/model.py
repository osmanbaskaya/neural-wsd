import logging
from abc import abstractmethod

from pytorch_transformers import BertForSequenceClassification, BertConfig

LOGGER = logging.getLogger(__name__)


class ExperimentBaseModel:
    _hparams = {}
    _tparams = {}

    def __init__(self):
        self.model = None

    @property
    def hparams(self):
        return self._hparams

    @property
    def tparams(self):
        return self._tparams

    def train(self, training_data, validation_data):
        LOGGER.info("train")

    def evaluate(self, test_data):
        pass

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError


class PretrainedExperimentModel(ExperimentBaseModel):
    _hparams = {"tokenizer": {"max_length": 512}, "model": {}, "runner": {"batch_size": 100}}
    _tparams = {"loader": {"shuffle": False, "num_workers": 1, "batch_size": 2}}

    def __init__(self, base_model, processor):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.num_labels = len(processor.label_encoder.classes_)

    def _get_model(self):
        bert_config = BertConfig.from_pretrained(self.base_model, num_labels=self.num_labels)
        return BertForSequenceClassification.from_pretrained(self.base_model, config=bert_config)
