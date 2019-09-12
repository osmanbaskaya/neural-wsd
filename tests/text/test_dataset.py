import pandas as pd
import pytest

from neural_wsd.text.dataset import WikiWordSenseDisambiguationDataset, sample_data


@pytest.fixture
def dataset():
    size = 11
    data = pd.DataFrame(
        {
            "text": ["here is a sentence"] * size,
            "sense": [0] * size,
            "offset": [2] * size,
            "word": ["here"] * size,
            "id": list(range(size)),
        }
    )
    return WikiWordSenseDisambiguationDataset(data=data)


def test_sample_data_size_correct(dataset):
    sample1, sample2 = sample_data(dataset, 0.5)
    assert len(sample1) == 6
    assert len(sample2) == 5


def test_sample_data_unique_samples(dataset):
    sample1, sample2 = sample_data(dataset, 0.5)
    assert len(set(sample1) & set(sample2)) == 0
