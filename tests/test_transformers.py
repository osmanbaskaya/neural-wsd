from neural_wsd.text.transformers import BasicTextTransformer, PaddingTransformer


def test_lowercase_op():
    t = BasicTextTransformer(name="text-prepocess", lowercase=True)
    assert t.transform(["Here some Text", "And More"])[0] == ["here some text", "and more"]


def test_padding_op_correct():
    t = PaddingTransformer(name="padding-op", max_length=5)
    truth = [[1, 2, 3, 0, 0], [1, 0, 5, 4, 0]]
    output, _ = t.transform([[1, 2, 3], [1, 0, 5, 4]])
    assert truth == output.tolist()
    assert output.shape == (2, 5)

