import torch
from pytorch_transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- Feature Transformers -------------------
from neural_wsd.text.dataset import WikiWordSenseDisambiguationDataset
from neural_wsd.text.ops import (
    PreTrainedModelTokenizeOp,
    BasicTextProcessingOp,
    PipelineRunner,
    PaddingOp,
)

class PreTrainedNeuralDisambiguator:
    base_model: str

    def __init__(self, base_model="bert-base-uncased", num_labels=2):
        self.config = BertConfig.from_pretrained(base_model, num_labels=num_labels)
        self.model = None
        self.data_pipeline = None
        self.base_model = base_model

    def train(self, data_loader):
        self.data_pipeline = self._get_data_pipeline()
        self.model = self._get_model()

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            pass

        d = self.data_pipeline.fit_transform(data)
        X, mask = d["default"], d["mask"]

        self.model = self._get_model()

    def _get_data_pipeline(self):
        # Todo: many hardcoded stuff here.
        lowercase_op = BasicTextProcessingOp(op_name="text-prepocess", lowercase=True)
        tokenizer_op = PreTrainedModelTokenizeOp(
            op_name="bert-tokenizer", base_model=self.base_model, max_length=512
        )
        padding_op = PaddingOp(op_name="padding-op")

        runner = PipelineRunner(op_name="initialization-operator", num_process=1, batch_size=5)

        # Connect the dots... Create the pipeline
        runner | lowercase_op | tokenizer_op | padding_op

        return runner

    def _get_model(self):
        return BertForSequenceClassification.from_pretrained(self.base_model, config=self.config)

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        d = self.data_pipeline.fit_transform(texts)
        X, attention_mask = d["default"], d["mask"]

        logits = self.model(
            input_ids=torch.tensor(X, dtype=torch.int64), attention_mask=attention_mask
        )

        return torch.argmax(logits)


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, do_basic_tokenize=False
)

max_length = 512

dataset = WikiWordSenseDisambiguationDataset(directory="dataset")
data_loader = DataLoader(dataset, 5, shuffle=False, num_workers=1)
batches = list(data_loader)

model = PreTrainedNeuralDisambiguator(
    base_model="bert-base-uncased", num_labels=dataset.num_of_unique_labels
)
model.train(data_loader)
model.predict(["here is some data", "some more"])
print(model)
