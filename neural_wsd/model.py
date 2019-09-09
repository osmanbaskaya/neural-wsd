from pytorch_transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from torch.utils.data import DataLoader

# ---------------- Feature Transformers -------------------
from neural_wsd.text.dataset import Tokenize, Padding, WikiWordSenseDisambiguationDataset


# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")


class PreTrainedNeuralDisambiguator:
    def __init__(self, num_labels, base_model="bert-base-uncased"):
        self.config = BertConfig.from_pretrained(base_model, num_labels=num_labels)
        self.model = BertForSequenceClassification.from_pretrained(base_model, config=self.config)

    def predict(self, text):
        # preprocess
        # return torch.argmax(self.model(text))
        pass


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, do_basic_tokenize=False
)

max_length = 512
tokenize = Tokenize(tokenizer=tokenizer, max_length=max_length)
pad = Padding(max_length=max_length)
dataset = WikiWordSenseDisambiguationDataset("dataset", transforms=[tokenize, pad])
data_loader = DataLoader(dataset, 5, shuffle=False, num_workers=1)
batches = list(data_loader)
model = PreTrainedNeuralDisambiguator(dataset.num_of_unique_labels)
print(model)
