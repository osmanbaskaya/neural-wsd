# coding=utf-8
import pytest
from pytorch_transformers import AutoTokenizer

from neural_wsd.text.transformers import asciify
from neural_wsd.text.transformers import BasicTextTransformer
from neural_wsd.text.transformers import PaddingTransformer
from neural_wsd.text.transformers import WordpieceToTokenTransformer


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


@pytest.mark.parametrize("base_model", [("roberta-base")])
def test_wordpiece_to_token_correct(base_model):
    t = WordpieceToTokenTransformer(name="wordpiece-to-token", base_model=base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Long text
    sentences = [
        "Some strange text sssasd sdafds dfv vc a more strange",
        "Short sentence",
        "OneToken",
        "",
    ]
    encoded_ids = [tokenizer.encode(sentence) for sentence in sentences]
    _, context = t.transform(encoded_ids)

    t = context["wordpiece_to_token_list"]

    assert [
        (0,),
        (1,),
        (2,),
        (3, 4, 5, 6),
        (7, 8, 9),
        (10, 11),
        (12, 13),
        (14,),
        (15,),
        (16,),
    ] == t[0]

    assert [(0,), (1,)] == t[1]
    assert [(0, 1)] == t[2]
    assert [] == t[3]
