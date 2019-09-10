from neural_wsd.text.ops import BasicTextProcessingOp, PaddingOp


def test_lowercase_op():
    op = BasicTextProcessingOp(op_name="text-prepocess", lowercase=True)
    assert op.transform(["Here some Text", "And More"]) == ["here some text", "and more"]


def test_padding_op_correct():
    op = PaddingOp(op_name="padding-op", max_length=5)
    truth = [[1, 2, 3, 0, 0], [1, 0, 5, 4, 0]]
    output, _ = op.transform([[1, 2, 3], [1, 0, 5, 4]])
    assert truth == output.tolist()
    assert output.shape == (2, 5)

