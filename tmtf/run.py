__doc__ = """Training loop. Should be executable from the terminal/commandline with a config file."""

# TODO Argparse


def fit(model, config):
    """
    :type: model: Antipasti.netarchs.model
    :param model: Antipasti model.

    :type config: dict
    :param config: Configuration.
    """
    # TODO
    pass


def readconfigfile(path):
    """
    Read configuration file from path.

    :type path: str
    :param path: Path to configuration yaml file.
    """
    assert path.endswith('.yml'), "Configuration file must be a YAML file."
    # TODO
    pass


if __name__ == '__main__':
    pass
