import logging
from .utils import configure_logger

configure_logger()
LOGGER = logging.getLogger(__name__)


def run():
    from .experiment.model import PretrainedExperimentModel

    from .processor import load_data, WikiWordSenseDataProcessor, ProcessorFactory

    base_model = "bert-base-uncased"
    cache_dir = "cache/exp1"

    processor = ProcessorFactory.get_or_create(
        WikiWordSenseDataProcessor, cache_dir=cache_dir, base_model=base_model
    )

    dataset_directory = "dataset"

    cached_data_fn = "wsd-data.pkl"
    datasets = load_data(processor, dataset_directory, cache_dir, cached_data_fn)
    model = PretrainedExperimentModel(base_model, processor)
    model.train(datasets["tsv"], datasets["ts"])


if __name__ == "__main__":
    run()
