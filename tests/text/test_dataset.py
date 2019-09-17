import pandas as pd
import pytest

from neural_wsd.text.dataset import sample_data
from neural_wsd.text.dataset import WikiWordSenseDisambiguationDataset


@pytest.fixture
def dataset():
    size = 11
    data = [
        list(range(size)),  # id
        ["bank"] * size,  # target word
        [0] * size,  # offset
        [2] * size,  # label / sense
        ["here is a sentence"] * size,  # annotated sentence (<target>target_word<target>)
        ["here is a sentence"] * size,  # tokenized sentence
        ["here is a sentence"] * size,  # original sentence
    ]

    data = zip(*data)
    return WikiWordSenseDisambiguationDataset(data=data)


def test_sample_data_size_correct(dataset):
    sample1, sample2 = sample_data(dataset, 0.5)
    assert len(sample1) == 6
    assert len(sample2) == 5


def test_sample_data_unique_samples(dataset):
    sample1, sample2 = sample_data(dataset, 0.5)
    assert len(set(sample1) & set(sample2)) == 0
