__doc__ = """Training loop. Should be executable from the terminal/commandline with a config file."""

# TODO Argparse


def fit(model, env, config, verbose=True):
    """
    :type: model: Antipasti.netarchs.model
    :param model: Antipasti model.

    :type env: sim.Simulator
    :param env: Environment object.

    :type config: dict or str
    :param config: Configuration dictionary or path to a YAML configuration file.
    """

    # Make print function
    def _print(msg):
        if verbose:
            print(msg)

    # Read config
    if isinstance(config, str):
        config = readconfigfile(config)

    # Check model
    # Model must be compiled
    assert model.classifiertrainer is not None, "Model must be compiled."
    # Model must have 'Q' in its output dictionary
    assert 'Q' in model.classifiertrainer.outputs.keys(), "'Q' must be in model outputs."

    _print("[+] Tests passed.")

    # Prepare loop variables
    # Episode counter
    episodecount = 0
    # Iteration counter
    itercount = 0

    # Begin loop
    while True:
        # Update loop variables
        itercount += 1

        # Check if loop must be broken
        if itercount > config['maxitercount']:
            _print("[-] Maximum number of iterations reached, aborting training...")
            break

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
