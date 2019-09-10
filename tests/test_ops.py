from neural_wsd.text.ops import (
    PreTrainedModelTokenizeOp,
    BasicTextProcessingOp,
    PipelineRunner,
    PaddingOp,
)
from multiprocessing import Pool

runner = PipelineRunner(op_name="initialization-operator", num_process=1, batch_size=5)

lowercase_op = BasicTextProcessingOp(op_name="text-prepocess", lowercase=True)
tokenizer_op = PreTrainedModelTokenizeOp(
    op_name="bert-tokenizer", base_model="bert-base-uncased", max_length=512
)
padding_op = PaddingOp(op_name="padding-op")

# lowercase_op = LowerCaseOp("lowercase", next_op=tokenizer_op)

# Create the pipeline
runner | lowercase_op | tokenizer_op | padding_op

data = runner.fit_transform(["here is some data", "here more data", "Even More Data"])
print(runner)
print("\n", data)


def test_lowercase_op():
    op = BasicTextProcessingOp("text-prepocess", lowercase=True)
    assert op.transform(["Lowercase this data"]) == ["lowercase this data"]
