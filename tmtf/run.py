import net
import sim
import edb as edb_

import numpy as np

__doc__ = """Training loop. Should be made executable from the terminal/commandline with a config file."""

# TODO Argparse


def fit(models, env, edb, config, verbose=True):
    """
    :type: model: list or tuple
    :param models: Tuple of (Q-Network, Target-Network).

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

    # Read networks
    model, targetmodel = models

    # # Check models
    assert hasattr(targetmodel, 'baggage'), "Target model must have baggage with target parameter update function."
    assert "updatetargetparams" in targetmodel.baggage.keys(), "Target parameter update function 'updatetargetparams' " \
                                                               "must be in targetmodel's baggage."
    assert model.savedir is not None, "Model must have a place to save parameters."

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
        gameclock = 0

        _print("| New Game |")

        # Sample initial state from environment
        state = env.getstate()

        while not isterminal:
            # ----- [GET XP] -----
            # Q is a tensor which can be squeezed to a vector, which can then be argmax-ed to get an action.
            # Select an action (1 - eps) greedily
            if config['greed'] >= np.random.uniform(low=0., high=1.):
                # Sample Q from network
                Q = model.classifier(state)
                # Find action to take
                action = Q.squeeze().argmax()
                _print("| [T = {}] Q-Values: {} |".format(gameclock, Q.squeeze()))
                _print("| [T = {}] Taking action {} greedily. |".format(gameclock, action))
            else:
                action = np.random.randint(low=0, high=config['numactions'])
                _print("| [T = {}] Taking action {} randomly. |".format(gameclock, action))

            # See what the environment thinks about this action
            reward, newstate, isterminal = env.getresponse(action)
            # Log to experience database
            edb.log(state, action, reward, newstate, isterminal)
            # Set state to newstate
            state = newstate

            # ----- [LEARN FROM XP] -----
            # Fetch new batch from experience database
            x, yt = edb.batcherbatcher(targetnetwork=targetmodel, gamma=config['gamma'],
                                       batchsize=config['batchsize'])

            try:
                # Train on batch
                out = model.classifiertrainer(x, yt)
                # Update target network parameters
                targetmodel.baggage["updatetargetparams"](params=model.params,
                                                          decay=config['targetnetworkparamdecay'])

            except Exception as e:
                # Print diagnostics
                _print("[-] Exception raised while training model. Printing Diagnostics:")
                diagnostics = {'yt.shape': yt.shape,
                               'x.shape': x.shape,
                               'len(edb.db)': len(edb.db)}
                for diagitem, diagres in diagnostics.items():
                    _print(">>> {} = {}".format(diagitem, diagres))
                # Make a substitute out such that training could continue
                out = {'C': 'Error', 'L': 'Error', 'yt': yt}

            # Print
            echomsg = "| Cost: {C} || Loss: {L} || Target: {targ} |".format(C=out['C'], L=out['L'], targ=yt)
            _print(echomsg)
            _print("------------------------")

            # Increment counters
            gameclock += 1

        # Check if the game was won
        gamewon = env.getreward() == 1.
        _print("| [{}] Game {} won. |".format(*{True: ('+', 'was'), False: ('-', 'was not')}[gamewon]))
        _print("\n")
        # Save parameters if required
        if itercount % config['saveevery'] == 0:
            _print("[+] Saving model parameters...\n")
            model.save('--iter-{}-routine'.format(itercount))
            targetmodel.save('--iter-{}-routine'.format(itercount))

    return model


def readconfigfile(path):
    """
    Read configuration file from path.

    :type path: str
    :param path: Path to configuration yaml file.
    """
    assert path.endswith('.yml') or path.endswith('.save'), "Configuration file must be a YAML file or a Pickled " \
                                                            "dictionary object with '.save' extension."
    # TODO Parse config file
    pass


def main(configpath):
    """Pretend this function doesn't exist."""

    # FIXME: Debug and prettify.
    if not __debug__:
        return

    # ---Build Environment---
    # Build videoframes
    videopath = '/media/data/nrahaman/DeepTrack/Data/movies/1'
    vf = sim.VideoFrames(videopath)
    # Build track
    trackpath = '/home/nrahaman/Python/Antipasti/Projects/DeepTrack/tracks/1/ctrax_results.mat'
    tr = sim.Track(trackpath)
    # Build Sim
    env = sim.FlySimulator(vf, tr)

    # ---Build EDB---
    ed = edb_.ExperienceDB(maxsize=100)

    # ---Build models---
    models = net.simple()

    savedirmodel = '/media/data/nrahaman/NotNeuro/tracky/Try1/Value/Weights/'
    savedirtarget = '/media/data/nrahaman/NotNeuro/tracky/Try1/Target/Weights/'
    models[0].savedir = savedirmodel
    models[1].savedir = savedirtarget

    # ---Train---
    # Build config
    config = {'batchsize': 1,
              'gamma': 0.9,
              'targetnetworkparamdecay': 0.9,
              'maxitercount': 1000000,
              'maxgamecount': 2,
              'saveevery': 500,
              'numactions': 5,
              'greed': 0.9}

    trainedmodel = fit(models, env, ed, config)
    trainedmodel.save('--final')


if __name__ == '__main__':
    main(None)
