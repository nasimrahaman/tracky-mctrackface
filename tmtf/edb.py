
__doc__ = """Experience Database for the DeepQ Learner."""

from random import randint
from Antipasti.netdatautils import pickle, unpickle
from numpy import inf

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

    def log(self, state, action, reward, newstate):
        # Append to database
        self.db.append((state, action, reward, newstate))
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
