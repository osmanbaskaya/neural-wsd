import yaml


def configure_logger():
    import logging.config

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
