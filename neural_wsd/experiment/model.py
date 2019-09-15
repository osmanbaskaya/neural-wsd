# coding=utf-8
import logging
from abc import abstractmethod
from collections import namedtuple

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm

from ..utils import merge_params

LOGGER = logging.getLogger(__name__)


class ExperimentBaseModel:
    def __init__(self, hparams=None, tparams=None):
        self.model = None

        self._hparams = merge_params(self.get_default_hparams(), hparams)
        self._tparams = merge_params(self.get_default_tparams(), tparams)

    @property
    def hparams(self):
        return namedtuple(f"{self.__class__.__name__}_hparams", sorted(self._hparams))(
            **self._hparams
        )

    @property
    def tparams(self):
        return namedtuple(f"{self.__class__.__name__}_tparams", sorted(self._tparams))(
            **self._tparams
        )

    def get_default_hparams(self):
        raise NotImplementedError()

    def get_default_tparams(self):
        raise NotImplementedError()

    def train(self, train_dataset, validation_dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tp = self.tparams
        hp = self.hparams
        model = self._get_model()

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=tp.batch_size
        )

        t_total = tp.max_steps
        num_train_epochs = tp.max_steps // len(train_dataloader) + 1

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": tp.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=tp.learning_rate, eps=tp.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=tp.warmup_steps, t_total=t_total)
        if tp.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(model, optimizer, opt_level=tp.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if tp.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # XLM and RoBERTa don't use segment_ids
                    "token_type_ids": batch[2]
                    if self.base_model_name() in ["bert", "xlnet"]
                    else None,
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                loss = outputs[
                    0
                ]  # model outputs are always tuple in pytorch-transformers (see doc)

                if tp.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if tp.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), tp.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tp.max_grad_norm)

                tr_loss += loss.item()
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if 0 < tp.max_steps < global_step:
                    epoch_iterator.close()
                    break
                if 0 < tp.max_steps < global_step:
                    train_iterator.close()
                    break

        return global_step, tr_loss / global_step

    def evaluate(self, test_data):
        pass

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError()

    def save(self):
        pass

    def load(self):
        pass

    def base_model_name(self):
        raise NotImplementedError()


class PretrainedExperimentModel(ExperimentBaseModel):
    _hparams = {"tokenizer": {"max_length": 512}, "model": {}, "runner": {"batch_size": 100}}
    _tparams = {"loader": {"shuffle": False, "num_workers": 1, "batch_size": 2}}

    def __init__(self, base_model, processor):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.num_labels = len(processor.label_encoder.classes_)

    def get_default_hparams(self):
        return {}

    def get_default_tparams(self):
        return {
            "batch_size": 1,
            "max_steps": 5,
            "weight_decay": 0.0,
            "adam_epsilon": 1e-8,
            "learning_rate": 5e-5,
            "fp16": False,
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
            "n_gpu": 1,
        }

    def _get_model(self):
        config = AutoConfig.from_pretrained(self.base_model, num_labels=self.num_labels)
        return AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config)

    def base_model_name(self):
        return self.base_model.split("-")[0]
