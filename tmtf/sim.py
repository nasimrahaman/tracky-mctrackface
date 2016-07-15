__doc__ = """Fly Tracking Environment for Q-Learning."""

import os
import glob

import numpy as np
from scipy.ndimage import imread, gaussian_filter

from Antipasti.netdatautils import ctrax2np


class Simulator(object):
    def __init__(self):
        pass

    def getstate(self):
        pass

    def getresponse(self, action):
        pass

    def setstate(self, state):
        pass

    def resetenv(self):
        pass


class FlySimulator(Simulator):
    def __init__(self, path, seed, framesperstate=4, episodelength=40):
        # Initialize superclass
        super(FlySimulator, self).__init__()

        # Assertions
        assert os.path.isdir(path), "Path must be a directory."
        assert framesperstate >= 1, "Must have at least 1 frame per step."

        # Meta
        self.path = path
        self.seed = seed
        self.framesperstate = framesperstate
        self.episodelength = episodelength

        # Internal dont-fuck-with-me attributes
        # Cross
        self._crosshair = None
        # Min global time
        self._minT = self.framesperstate - 1
        # Max global time
        self._maxT = len(self.filenames)
        # Training could occur in episodes which may not span the entire video sequence. In general, the tracking
        # problem doesn't define an episode (unlike in Atari games, where there's a game-start and game-over).
        # The environment should support arbitrary episodes as long as it's valid.
        self._episodeT = None
        self._episodestart = None
        self._episodestop = None

        # Parse directory
        self.filenames = None
        self.imshape = None
        self.parsedir()

        # State tuple
        self._state = None

        # Make a persistent crosshair and init with seed
        self.crosshair = self.seed

    @property
    def crosshair(self):
        return self._crosshair

    @crosshair.setter
    def crosshair(self, value):
        # Assertions on value
        # TODO
        # Set
        self._crosshair = value

    @property
    def episodeT(self):
        return self._episodeT

    @episodeT.setter
    def episodeT(self, value):
        assert value >= self._minT
        assert value <= self._maxT
        self._episodeT = value

    def initepisode(self, episodestart, episodestop):
        """Initialize an episode."""
        assert self._minT <= episodestart < episodestop <= self._maxT

        self.episodeT = self._episodestart = episodestart
        self._episodestop = episodestop

    def getstate(self):
        # Fetch frames
        frames = self.fetchframes()
        # Fetch crosshair image
        crossimg = self.crosshair_image(imshape=self.imshape, coordinates=self.crosshair)
        # TODO

    def resetenv(self):
        """Reset environment."""
        self.crosshair = self.seed

    def parsedir(self):
        # Filenames need to be sorted to get rid of the dictionary ordering
        self.filenames = sorted(glob.glob("{}/*.png".format(self.path)), key=lambda x: int(x.split('.')[0]))
        self.imshape = imread(self.filenames[0]).shape

    def fetchframes(self, stop=None, numsteps=None):
        """Fetch `numsteps` frames starting at `start` and concatenate along the 0-th axis."""

        if stop is None:
            stop = self.episodeT

        if numsteps is None:
            numsteps = self.framesperstate

        frames = np.array([imread(self.filenames[framenum])
                           for framenum in range(stop - numsteps + 1, stop + 1)])[::-1, ...]
        return frames

    @staticmethod
    def crosshair_image(imshape, coordinates, smooth=0):
        """
        Function to make a zero image of shape `imshape`, place a white (one) pixel or a gaussian blob at the given
        `coordinates`.
        """
        # make CROSShair IMaGe
        crossimg = np.zeros(shape=imshape)
        # Place a pixel
        crossimg[coordinates] = 1.
        # Smooth if required to
        if smooth != 0:
            crossimg = gaussian_filter(crossimg, sigma=smooth)
        # Done
        return crossimg


class Track(object):
    """Class to wrap ctrax MATLAB file."""
    def __init__(self, matfilepath, objectid):
        """
        :type matfilepath: str
        :param matfilepath: Path to matlab file.

        :type objectid: int
        :param objectid: ID of the object being tracked.
        """

        # Meta
        self.matfilepath = matfilepath
        self.objectid = objectid

        # Parse matfile to a position array
        self.posarray = ctrax2np(matpath=matfilepath)

        assert self.objectid < self.posarray.shape[1], "Object ID does not correspond to an object in the ctrax file."

    def getposition(self, T):
        return self.posarray[T, self.objectid, :]

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.stop < self.posarray.shape[0]
            assert item.start > 0
        elif isinstance(item, int):
            assert item < self.posarray.shape[0]
            assert item > 0
        else:
            raise NotImplementedError("`item` must be a slice or an integer.")

        # Get position
        pos = self.getposition(item)

        # Convert to a tuple if pos is just a 2-vector and return
        if pos.ndim == 1:
            return tuple(pos)
        else:
            return pos
