# coding=utf-8
import logging

import torch
from pytorch_transformers import RobertaForSequenceClassification
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from .base import PretrainedExperimentModel

LOGGER = logging.getLogger()


class RobertaBaseModel(RobertaForSequenceClassification):
    def __init__(self, config, classifier):
        super().__init__(config)
        self.classifier = classifier

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        wordpiece_to_token_list=None,
        offsets=None,
    ):
        outputs = self.roberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(
            sequence_output, wordpiece_to_token_list=wordpiece_to_token_list, offsets=offsets
        )

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


#  ----- Classification/Extraction Heads ------


class TokenEmbedding(nn.Module):
    def __init__(self, weighted_cls_token=True):
        super().__init__()
        self.weighted_cls_token = weighted_cls_token
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sequence_output, wordpiece_to_token_list, **kwargs):
        # TODO remove `assert` this when you have unit test.
        assert len(sequence_output) == len(wordpiece_to_token_list)

        batch_size, max_length, hidden_size = sequence_output.shape

        output = torch.zeros_like(sequence_output)

        # Obviously this is very inefficient. Is there any way to do this in one pass without
        # blowing up memory?
        for i in range(batch_size):
            for j, word_piece_indices in enumerate(wordpiece_to_token_list[i]):
                output[i, j, :] = sequence_output[i, word_piece_indices, :].sum(0)

        return output


class RobertaTokenModel(PretrainedExperimentModel):
    def __init__(self, base_model, processor, device=None, hparams=None, tparams=None):
        super().__init__(base_model, processor, device, hparams, tparams)

    def _get_model(self):
        config = self._get_config()
        classifier = MultiSequential(TokenEmbedding(), RobertaTokenClassificationHead(config))
        return RobertaBaseModel(config, classifier)

    def get_additional_training_params(
        self, dataset, train_loader, validation_loader, train_sampler, validation_sampler
    ):
        train_alignments_loader = DataLoader(
            dataset.alignments,
            self.tparams["batch_size"],
            sampler=train_sampler,
            collate_fn=lambda t: t,
        )
        valid_alignments_loader = DataLoader(
            dataset.alignments,
            self.tparams["batch_size"] * 2,
            sampler=validation_sampler,
            collate_fn=lambda t: t,
        )

        return {
            "train_alignments_loader": train_alignments_loader,
            "validation_alignment_loader": valid_alignments_loader,
        }

    def _prepare_batch_input(self, batch):

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "offsets": batch[2]}

        if len(batch) == 4:
            inputs["labels"] = batch[3]

        return {k: v.to(self.device) for k, v in inputs.items()}

    def _train_loop(self, data_loader, optimizer, scheduler=None, **kwargs):
        self.model.train()
        alignment_loader = kwargs["train_alignments_loader"]
        total_loss = 0
        num_step = 0
        accuracy = 0
        for num_step, batch in enumerate(zip(data_loader, alignment_loader), 1):
            input_batch, wordpiece_batch = batch
            inputs = self._prepare_batch_input(input_batch)
            outputs = self.model(**inputs, wordpiece_to_token_list=wordpiece_batch)
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

        alignment_loader = kwargs["validation_alignment_loader"]
        for num_step, batch in enumerate(zip(data_loader, alignment_loader), 1):
            input_batch, wordpiece_batch = batch
            inputs = self._prepare_batch_input(input_batch)
            outputs = self.model(**inputs, wordpiece_to_token_list=wordpiece_batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            accuracy += self.accuracy(logits, inputs["labels"]).item()

        return total_loss, accuracy / len(data_loader)

    def _predict(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]

        features, _ = self.processor.transform(sentences)

        # add offset information for prediction. If offset is not provided, the model will use
        # index 0 (i.e. [CLS]) to predict sense of the token. This may give inaccurate results.
        if "offset" in kwargs:
            offsets = kwargs["offset"]
            if isinstance(offsets, int):
                offsets = [offsets]
            features["offset"] = offsets

        features = self.processor.create_examples(features)

        feature_dataset = self.processor.create_tensor_data(features)
        dl = DataLoader(
            feature_dataset.tensor_dataset, self.tparams["batch_size"] * 2, shuffle=False
        )
        alignment_loader = DataLoader(
            feature_dataset.alignments,
            self.tparams["batch_size"] * 2,
            shuffle=False,
            collate_fn=lambda t: t,
        )

        tensors = []
        with torch.no_grad():
            for input_batch, wordpiece_batch in zip(dl, alignment_loader):
                inputs = self._prepare_batch_input(input_batch)
                tensors.append(self.model(**inputs, wordpiece_to_token_list=wordpiece_batch)[0])

        return torch.cat(tensors=tensors).cpu().numpy()


class MultiSequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self._modules.values():
            input = module(input, **kwargs)
        return input


class RobertaTokenClassificationHead(RobertaClassificationHead):
    """Head for token-level classification."""

    def forward(self, features, **kwargs):
        offsets = kwargs["offsets"]
        x = features[torch.arange(features.size(0)), offsets, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
