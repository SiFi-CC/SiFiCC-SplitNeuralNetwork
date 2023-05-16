import os
import numpy as np
import spektral

from spektral.data import Dataset


class SiFiCCNNDense():
    def __init__(self, name, dataformat, n=None, **kwargs):
        self.name = name
        self.dataformat = dataformat
        self.n = n

        super().__init__(**kwargs)

    @property
    def path(self):
        # get current path, go two subdirectories higher
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.dirname(path)
        path = os.path.dirname(path)

        return os.path.join(path,
                            self.__class__.__name__,
                            self.name)


    def download(self):
        print("Dunno some download function")
        print(self.path)

    def read(self):
        print("Dunno some read function")
