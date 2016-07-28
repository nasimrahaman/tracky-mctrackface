
__doc__ = """Experience Database for the DeepQ Learner."""

from random import randint
from Antipasti.netdatautils import pickle, unpickle
from numpy import inf
import numpy as np


class ExperienceDB(object):
    """Experience Database for DQNs."""
    def __init__(self, filename=None, maxsize=None):
        """
        Initializes an experience database and binds it to a file.

        :type filename: str
        :param filename: Name of the file to bind the ExperienceDB to.

        :type maxsize: int
        :param maxsize: Maximum size of the database. Defaults to infinity when
                        omitted, proceed with caution.
        """
        # Initialize database to an empty list (no fancy datastructure yet, but
        # this needs to be replaced with a memory mapped list.)
        self.db = []
        # Filename to bind
        self.filename = filename
        # Maximum database size
        self.maxsize = inf if maxsize is None else maxsize

    def log(self, state, action, reward, newstate, isterminal):
        # Append to database
        self.db.append((state, action, reward, newstate, isterminal))
        # If database bigger than maxsize, pop the first entry
        if len(self.db) > self.maxsize:
            self.db.pop(0)

    def fetch(self, expidx=None):
        """
        Fetch (random) experience from the database.

        :type expidx: int
        :param expidx: Experience ID (if any given). Fetch random if omitted.
        """
        if expidx is None:
            expidx = randint(0, len(self.db) - 1)
        return self.db[expidx]

    # batcher batcher batcher MUSHROOM MUSHROOM
    def batcher(self, targetnetwork, gamma):
        """Sample from experience database and make a batch."""
        # Fetch from experience database at random
        state, action, reward, newstate, isterminal = self.fetch()
        # Compute new Q with target network
        newQ = targetnetwork.classifier(newstate)
        # Compute target
        if not isterminal:
            reward = 0 if reward is None else reward
            target = reward + gamma * np.max(newQ)
        else:
            target = reward

        return state, target

    def batcherbatcher(self, targetnetwork, gamma, batchsize=1):
        """Sample from experience database multiple times and make a large batch."""
        # FIXME: Batchsize = 1 for now.
        assert batchsize == 1, "Batch-size must be 1 for now, though this is easily fixed (Read: Nasim is a lazy arse)."
        # Call batcher.
        return self.batcher(targetnetwork, gamma)

    def _write(self, filename=None):
        """
        Write database to file (for persistent backup).

        :type filename: str
        :param filename: Where to write. Will use default directory if omitted.
        """
        # self.db could be a mmap object, so make sure it isn't one.
        if not isinstance(self.db, list):
            return

        if filename is None:
            filename = self.filename
        else:
            # Update filename
            self.filename = filename

        # Pickle
        pickle(self.db, filename)

    def _read(self, filename=None):
        """
        Read from database.

        :type filename: str
        :param filename: Where to read from. Will use default directory if
                         omitted.
        """

        if filename is None:
            filename = self.filename
        else:
            # Update filename
            self.filename = filename
        # Read
        self.db = unpickle(filename)
