# coding=utf-8
from neural_wsd.text.transformers import asciify
from neural_wsd.text.transformers import BasicTextTransformer
from neural_wsd.text.transformers import PaddingTransformer


def test_lowercase_op():
    t = BasicTextTransformer(name="text-prepocess", to_lowercase=True)
    assert t.transform(["Here some Text", "And More"])[0] == ["here some text", "and more"]


def test_padding_op_correct():
    t = PaddingTransformer(name="padding-op", max_seq_len=5)
    truth = [[1, 2, 3, 0, 0], [1, 0, 5, 4, 0]]
    output, _ = t.transform([[1, 2, 3], [1, 0, 5, 4]])
    assert truth == output.tolist()
    assert output.shape == (2, 5)


def test_asciify_correct():
    # Todo (kerem, osman): add more text cases, especially for English.
    assert asciify("Ślusàrski") == "Slusarski"
    assert asciify("kierowców") == "kierowcow"
    assert (
        asciify("Sıfır noktasındayız. Olayın şerefine bir konuşma yapacak mısın?") == "Sifir "
        "noktasindayiz. Olayin serefine bir konusma yapacak misin?"
    )
    assert (
        asciify("Here is some text that shouldn't be changed.") == "Here is some text that "
        "shouldn't be changed."
    )
