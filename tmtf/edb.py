
__doc__ = """Experience Database for the DeepQ Learner."""

class ExperienceDB(object):
    def __init__(self):
        pass

    def log(self, state, action, reward, newstate):
        pass

    def fetch(self, expidx=None):
        """
        Fetch experience from the database.

        :type expidx: int
        :param expidx: Experience ID (if any given). Fetch random if omitted.
        """
        pass

    def _write(self, filename=None):
        """
        Write database to file (for persistent backup).

        :type filename: str
        :param filename: Where to write. Use default directory if omitted.
        """
        pass

    def _read(self, filename=None):
        """
        Read from database.
        """
        pass
