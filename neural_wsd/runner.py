import logging

from .experiment.model import PretrainedExperimentModel
from .processor import load_data
from .processor import ProcessorFactory
from .processor import WikiWordSenseDataProcessor
from .utils import configure_logger

configure_logger()
LOGGER = logging.getLogger(__name__)
BASE_MODEL = "distilbert-base-uncased"

cache_dir = "cache/exp1"
cached_data_fn = "wsd-data.pkl"
dataset_directory = "dataset"


def create_model_processor(max_seq_len, base_model):
    processor_params = {"hparams": {"tokenizer": {"max_seq_len": max_seq_len}}}
    processor = ProcessorFactory.get_or_create(
        WikiWordSenseDataProcessor, cache_dir=cache_dir, base_model=base_model, **processor_params
    )
    LOGGER.info(f"{processor.hparams}")
    LOGGER.info(f"{processor.tparams}")
    return processor


def get_data(processor):
    datasets = load_data(processor, dataset_directory, cache_dir, cached_data_fn)
    return datasets


def get_model(processor, base_model, tparams):
    model = PretrainedExperimentModel(base_model, processor, tparams=tparams)
    return model


def run(max_seq_len, base_model, tparams):
    processor = create_model_processor(max_seq_len=max_seq_len, base_model=base_model)
    datasets = get_data(processor)

    model = get_model(processor, base_model, tparams)
    global_step, training_loss = model.train(datasets["tsv"])

    sentences = ["Bass likes warm waters.", "Bass music is great"]

    print(model.predict(sentences))

    model.save()

    return model, datasets, global_step, training_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--max-seq-len", default=64, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--base-model", default=BASE_MODEL, type=str)
    args = parser.parse_args()

    LOGGER.info(f"{args}")

    tparams = {"batch_size": args.batch_size, "max_steps": args.max_steps}

    run(args.max_seq_len, args.base_model, tparams)


if __name__ == "__main__":
    main()
