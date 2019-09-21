# coding=utf-8
import torch
from pytorch_transformers import RobertaForSequenceClassification
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

from .base import PretrainedExperimentModel


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
    ):
        outputs = self.roberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, wordpiece_to_token_list=wordpiece_to_token_list)

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

    def forward(self, sequence_output, wordpiece_to_token_list):
        assert len(sequence_output) == len(wordpiece_to_token_list)

        batch_size, max_length, hidden_size = sequence_output.shape

        output = torch.zeros_like(sequence_output)

        # Obviously this is very inefficient. Is there any way to do this in one pass without
        # blowing up memory?
        for i in range(batch_size):
            for j, word_piece_indices in enumerate(wordpiece_to_token_list[i]):
                output[i, j, :] = (sequence_output[i, word_piece_indices, :]).sum()
        return output


class RobertaTokenModel(PretrainedExperimentModel):
    def __init__(self, base_model, processor, device=None, hparams=None, tparams=None):
        super().__init__(base_model, processor, device, hparams, tparams)

    def _get_model(self):
        config = self._get_config()
        classifier = MultiSequential(TokenEmbedding(), RobertaClassificationHead(config))
        return RobertaBaseModel(config, classifier)

    def _prepare_batch_input(self, batch):

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

        # XLM and RoBERTa don't use segment_ids
        if self.base_model_arch in ["bert", "xlnet"]:
            inputs["token_type_ids"] = batch[2]

        if len(batch) == 4:
            inputs["labels"] = batch[3]

        return {k: v.to(self.device) for k, v in inputs.items()}


class MultiSequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self._modules.values():
            input = module(input, **kwargs)
        return input
