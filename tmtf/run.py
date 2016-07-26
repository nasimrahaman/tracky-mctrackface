__doc__ = """Training loop. Should be executable from the terminal/commandline with a config file."""

# TODO Argparse


def fit(model, env, edb, config, verbose=True):
    """
    :type: model: Antipasti.netarchs.model
    :param model: Antipasti model.

    :type env: sim.Simulator
    :param env: Environment object.

    :type edb: edb.ExperienceDB
    :param edb: Experience database.

    :type config: dict or str
    :param config: Configuration dictionary or path to a YAML configuration file.

    :type verbose: str
    :param verbose: quack quack.
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
    itercount = -1
    # Epoch counter
    gamecount = 0

    # Begin loop
    while True:
        # Update loop variables
        itercount += 1

        # Check if loop must be broken
        if itercount > config['maxitercount']:
            _print("[-] Maximum number of iterations reached, aborting training...")
            break

        if gamecount > config['maxgamecount']:
            _print("[-] Maximum number of epochs reached, aborting training...")
            break

        # Start new game
        env.newgame()
        gamecount += 1
        isterminal = False

        # Sample initial state from environment
        state = env.getstate()

        while not isterminal:
            # ----- [GET XP] -----
            # Sample from network
            Q = model.classifier(state)
            # Q is a tensor which can be squeezed to a vector, which can then be argmax-ed to get an action.
            action = Q.squeeze().argmax()
            # See what the environment thinks about this action
            reward, newstate, isterminal = env.getresponse(action)
            # Log to experience database
            edb.log(state, action, reward, newstate, isterminal)

            # ----- [LEARN FROM XP] -----
            # TODO
            pass
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
