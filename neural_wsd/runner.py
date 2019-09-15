import logging
from .utils import configure_logger

configure_logger()
LOGGER = logging.getLogger(__name__)


def run():
    from .experiment.model import PretrainedExperimentModel

    from .processor import load_data, WikiWordSenseDataProcessor, ProcessorFactory

    # base_model = "bert-base-uncased"
    base_model = "distilbert-base-uncased"
    cache_dir = "cache/exp1"

    processor_params = {"hparams": {"tokenizer": {"max_seq_len": 64}}}
    processor = ProcessorFactory.get_or_create(
        WikiWordSenseDataProcessor, cache_dir=cache_dir, base_model=base_model, **processor_params
    )

    print(processor.hparams)
    print(processor.tparams)

    dataset_directory = "dataset"

    cached_data_fn = "wsd-data.pkl"
    datasets = load_data(processor, dataset_directory, cache_dir, cached_data_fn)

    tparams = {"batch_size": 320}

    model = PretrainedExperimentModel(base_model, processor, tparams=tparams)
    model.train(datasets["tsv"], datasets["ts"])
    print(model.predict(["Bass likes warm waters.", "Bass music is great"]))


if __name__ == "__main__":
    run()
