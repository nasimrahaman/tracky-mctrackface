__doc__ = """Fly Tracking Environment for Q-Learning."""

import os
import glob

import numpy as np
from scipy.ndimage import imread, gaussian_filter


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

        # Parse directory
        self.filenames = None
        self.imshape = None
        self.parsedir()

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

    def initepisode(self):
        """Initialize an episode."""
        pass

    def getstate(self):
        # Fetch frames
        pass

    def resetenv(self):
        """Reset environment."""
        self.crosshair = self.seed

    def parsedir(self):
        # Filenames need to be sorted to get rid of the dictionary ordering
        self.filenames = sorted(glob.glob("{}/*.png".format(self.path)), key=lambda x: int(x.split('.')[0]))
        self.imshape = imread(self.filenames[0]).shape

    def fetchframes(self, stop, numsteps):
        """Fetch `numsteps` frames starting at `start` and concatenate along the 0-th axis."""
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
