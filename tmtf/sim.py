__doc__ = """Fly Tracking Environment for Q-Learning."""

import os
import glob

import Antipasti.pykit as pyk

import numpy as np


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
    def __init__(self, path):
        # Initialize superclass
        super(FlySimulator, self).__init__()

        # Assertions
        assert os.path.isdir(path), "Path must be a directory."

        # Meta
        self.path = path

        # Parse directory
        self.filenames = None
        self.maxT = None
        self.minT = None
        self.parsedir()

        # Make a global clock
        self.T = self.minT

    def getstate(self):
        pass

    def resetenv(self):
        self.T = 0

    def parsedir(self):
        self.filenames = sorted(glob.glob("{}/*.png".format(self.path)), key=lambda x: int(x.split('.')[0]))
        self.maxT = int(self.filenames[-1].split('.')[0])
        self.minT = int(self.filenames[0].split('.')[0])


