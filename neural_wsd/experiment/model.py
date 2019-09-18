# coding=utf-8
import logging
from abc import abstractmethod

import dill
import torch
from pytorch_transformers import AdamW
from pytorch_transformers import AutoConfig
from pytorch_transformers import AutoModelForSequenceClassification
from pytorch_transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.auto import trange

from ..text.dataset import sample_data
from ..utils import merge_params
from ..utils import total_num_of_params

LOGGER = logging.getLogger(__name__)


class ExperimentBaseModel:
    def __init__(self, processor, device=None, hparams=None, tparams=None):
        self.model = None
        self.processor = processor
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._hparams = merge_params(self.get_default_hparams(), hparams)
        self._tparams = merge_params(self.get_default_tparams(), tparams)

    @property
    def hparams(self):
        return self._hparams

    @property
    def tparams(self):
        return self._tparams

    def get_default_hparams(self):
        raise NotImplementedError()

    def get_default_tparams(self):
        raise NotImplementedError()

    def _prepare_batch_input(self, batch):

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

        # XLM and RoBERTa don't use segment_ids
        if self.base_model_arch in ["bert", "xlnet"]:
            inputs["token_type_ids"] = batch[2]

        if len(batch) == 4:
            inputs["labels"] = batch[3]

        return {k: v.to(self.device) for k, v in inputs.items()}

    @staticmethod
    def get_params_to_optimize(model, weight_decay, optimize_pretrained_model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            }
        ]

        if optimize_pretrained_model:
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                }
            )

        return optimizer_grouped_parameters

    def train(self, train_dataset):
        LOGGER.info(f"Training will be on: {self.device}")

        tp = self.tparams

        self.model = self._get_model().to(self.device)

        LOGGER.debug(self.model)
        total_params = total_num_of_params(self.model.named_parameters())
        LOGGER.info(f"Total number of params for {self.base_model_arch}: {total_params}")

        train_sampler, validation_sampler = sample_data(train_dataset, 0.9, shuffle=False)
        train_dataloader = DataLoader(train_dataset, tp["batch_size"], sampler=train_sampler)
        validation_sampler = DataLoader(train_dataset, tp["batch_size"], sampler=validation_sampler)

        t_total = self.tparams["max_steps"]
        num_train_epochs = tp["max_steps"] // len(train_dataloader) + 1

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_params_to_optimize(
            self.model, tp["weight_decay"], tp["optimize_pretrained_model"]
        )

        optimizer = self.get_optimizer(optimizer_grouped_parameters)
        scheduler = self.get_scheduler(optimizer, t_total, tp["warmup_steps"])

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        for _ in trange(num_train_epochs, desc="epoch"):
            self.model.train()
            loss_for_epoch, num_of_steps = self._train_loop(train_dataloader, optimizer, scheduler)
            # self.model.eval()
            # loss_for_epoch, num_of_steps = self._train_loop(train_dataloader, optimizer, scheduler)

        return global_step, tr_loss / global_step

    def get_optimizer(self, grouped_params):
        optimizer = self.tparams["optimizer"]["optimizer"]
        opt_params = self.tparams["optimizer"]["params"]
        return optimizer(grouped_params, **opt_params)

    def get_scheduler(self, optimizer, t_total, warmup_steps):
        return WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    def _train_loop(self, dataloader, optimizer=None, scheduler=None):
        total_loss = 0
        num_of_steps = 0
        for step, batch in enumerate(tqdm(dataloader, desc="batch")):
            inputs = self._prepare_batch_input(batch)
            outputs = self.model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tparams["max_grad_norm"])
            total_loss += loss.item()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.model.zero_grad()
            num_of_steps += 1
        return total_loss, num_of_steps

    def predict(self, sentences):
        predictions = self._predict(sentences)
        return self.processor.inverse_transform_labels(predictions.argmax(axis=1))

    def _predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        data = self.processor.transform(sentences)
        with torch.no_grad():
            self.model.eval()
            tensor_data = self.processor.create_tensor_data(data, labels_available=False)
            pred_iter = map(
                self._prepare_batch_input, DataLoader(tensor_data, self.tparams.batch_size)
            )
            return torch.cat(tensors=[self.model(**batch)[0] for batch in pred_iter]).cpu().numpy()

    def evaluate(self, test_data):
        pass

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError()

    def save(self):
        torch.save(self.model, "model.pkl", pickle_module=dill)

    def load(self):
        pass

    @property
    def base_model_arch(self):
        raise NotImplementedError()


class PretrainedExperimentModel(ExperimentBaseModel):
    def __init__(self, base_model, processor, device=None, hparams=None, tparams=None):
        super().__init__(processor, device, hparams, tparams)
        self.base_model = base_model
        self.processor = processor
        self.num_labels = len(processor.label_encoder.classes_)

    def get_default_hparams(self):
        return {"max_seq_len": 128}

    def get_default_tparams(self):
        return {
            "batch_size": 512,
            "max_steps": 100,
            "weight_decay": 0.0,
            "fp16": False,
            "fp16_opt_level": "O1",
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
            "n_gpu": 1,
            "optimize_pretrained_model": False,
            "optimizer": {"optimizer": AdamW, "params": {"lr": 5e-5, "eps": 1e-8}},
        }

    def _get_model(self):
        config = AutoConfig.from_pretrained(self.base_model, num_labels=self.num_labels)
        return AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config)

    @property
    def base_model_arch(self):
        return self.base_model.split("-")[0]
