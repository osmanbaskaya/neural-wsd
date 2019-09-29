# coding=utf-8
import logging
from abc import abstractmethod

import dill
import numpy as np
import torch
from pytorch_transformers import AdamW
from pytorch_transformers import AutoConfig
from pytorch_transformers import AutoModelForSequenceClassification
from pytorch_transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..text.dataset import FeatureDataset
from ..text.dataset import sample_data
from ..utils import merge_params
from ..utils import total_num_of_params

LOGGER = logging.getLogger(__name__)
LOGGER.disabled = False


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

    def accuracy(self, logits, gold_labels):
        correct = (logits.argmax(1) == gold_labels).sum().float()
        return correct / gold_labels.size(0)

    def _prepare_batch_input(self, batch):
        raise NotImplementedError()

    def create_dataloaders(self, dataset):
        train_sampler, validation_sampler = sample_data(dataset, 0.9, shuffle=True)
        train_dataloader = DataLoader(dataset, self.tparams["batch_size"], sampler=train_sampler)
        validation_loader = DataLoader(
            dataset, self.tparams["batch_size"] * 2, sampler=validation_sampler
        )

        return train_dataloader, validation_loader, train_sampler, validation_sampler

    def get_additional_training_params(
        self, dataset, train_loader, validation_loader, train_sampler, validation_sampler
    ):
        """This function should be overridden in subclasses to customize training."""
        return {}

    @staticmethod
    @abstractmethod
    def get_params_to_optimize(model, weight_decay, optimize_pretrained_model):
        raise NotImplementedError()

    def train(self, dataset: FeatureDataset):
        LOGGER.info(f"Training will be on: {self.device}")

        tp = self.tparams

        self.model = self._get_model().to(self.device)

        LOGGER.debug(self.model)
        total_params = total_num_of_params(self.model.named_parameters())
        LOGGER.info(f"Total number of params for {self.base_model_arch}: {total_params}")

        train_loader, validation_loader, train_sampler, validation_sampler = self.create_dataloaders(
            dataset.tensor_dataset
        )

        additional_params = self.get_additional_training_params(
            dataset, train_loader, validation_loader, train_sampler, validation_sampler
        )

        t_total = self.tparams["max_steps"]
        num_train_epochs = tp["max_steps"] // len(train_loader) + 1

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_params_to_optimize(
            self.model, tp["weight_decay"], tp["optimize_pretrained_model"]
        )

        optimizer = self.get_optimizer(optimizer_grouped_parameters)
        scheduler = self.get_scheduler(optimizer, t_total)

        global_step = 0
        tr_loss = 0.0
        best_validation_loss = np.inf
        patient = 0

        progress = tqdm(total=t_total, desc="Epoch")
        for epoch in range(1, num_train_epochs + 1):
            loss_for_epoch, num_step, accuracy = self._train_loop(
                train_loader, optimizer, scheduler=scheduler, **additional_params
            )

            LOGGER.info(f"Epoch accuracy: {accuracy}")
            print(f"Epoch accuracy: {accuracy}")

            global_step += num_step
            tr_loss += loss_for_epoch

            progress.update(num_step)

            # TODO Checkpoint logic is missing.
            if epoch % self.tparams["evaluate_every_n_epoch"] == 0:
                valid_set_loss, validation_accuracy = self._eval_loop(
                    validation_loader, **additional_params
                )
                print(f"Validation Epoch accuracy: {validation_accuracy}")
                if best_validation_loss > valid_set_loss:
                    LOGGER.info(
                        f"Validation loss is decreased from {best_validation_loss} to "
                        f"{valid_set_loss}"
                    )
                    patient = 0
                    best_validation_loss = valid_set_loss
                else:
                    LOGGER.info(
                        f"Validation loss didn't improved. T best validation loss so far is:"
                        f" {best_validation_loss} and the last validation loss is: {valid_set_loss}"
                    )
                    patient += 1

                # Early stopping.
                if self.tparams["patient"] <= patient:
                    # idea: copy the most recent model to another directory.
                    LOGGER.info(
                        f"Early stopping with patient: {patient} and with best valid "
                        f"score {best_validation_loss}"
                    )
                    break

        return global_step, tr_loss / global_step

    def get_optimizer(self, grouped_params):
        optimizer = self.tparams["optimizer"]["optimizer"]

        opt_params = self.tparams["optimizer"]["params"]
        return optimizer(grouped_params, **opt_params)

    def get_scheduler(self, optimizer, t_total):
        return WarmupLinearSchedule(
            optimizer, warmup_steps=self.tparams["warmup_steps"], t_total=t_total
        )

    def _train_loop(self, data_loader, optimizer, scheduler=None, **kwargs):
        self.model.train()
        total_loss = 0
        num_step = 0
        accuracy = 0
        for num_step, batch in enumerate(data_loader, 1):
            inputs = self._prepare_batch_input(batch)
            outputs = self.model(**inputs)
            loss, logits = outputs[:2]
            accuracy += self.accuracy(logits, inputs["labels"]).item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tparams["max_grad_norm"])
            total_loss += loss.item()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            self.model.zero_grad()

        return total_loss, num_step, accuracy / len(data_loader)

    def _eval_loop(self, data_loader, **kwargs):
        self.model.eval()
        total_loss = 0.0
        accuracy = 0
        for num_step, batch in enumerate(data_loader, 1):
            inputs = self._prepare_batch_input(batch)
            loss, logits = self.model(**inputs)[:2]
            total_loss += loss.item()
            accuracy += self.accuracy(logits, inputs["labels"]).item()

        return total_loss, accuracy / len(data_loader)

    def predict(self, sentences):
        self.model.eval()
        predictions = self._predict(sentences)
        return self.processor.inverse_transform_labels(predictions.argmax(axis=1))

    def _predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        data = self.processor.transform(sentences)
        with torch.no_grad():
            tensor_data = self.processor.create_tensor_data(data, labels_available=False)
            pred_iter = map(
                self._prepare_batch_input, DataLoader(tensor_data, self.tparams["batch_size"])
            )
            return torch.cat(tensors=[self.model(**batch)[0] for batch in pred_iter]).cpu().numpy()

    def evaluate(self, test_data):
        """
        :param test_data:
        :return: average loss.
        """
        data_loader = DataLoader(test_data, self.tparams["batch_size"], shuffle=False)
        total_loss = self._eval_loop(data_loader)
        return total_loss / len(test_data)

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
        print(self.tparams)
        print(self.hparams)

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
            "evaluate_every_n_epoch": 2,
            "patient": 10,
            "optimizer": {"optimizer": AdamW, "params": {"lr": 5e-5, "eps": 1e-8}},
        }

    def _get_config(self):
        config = AutoConfig.from_pretrained(self.base_model, num_labels=self.num_labels)
        return config

    def _get_model(self):
        config = self._get_config()
        return AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config)

    @property
    def base_model_arch(self):
        return self.base_model.split("-")[0]

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
                "weight_decay": weight_decay,
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
                    "weight_decay": 0.0,
                }
            )

        return optimizer_grouped_parameters
