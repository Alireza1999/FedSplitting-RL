from Tensorforce import config
import numpy as np


class NoSplitting:

    def __init__(self, environment):
        self.deviceNum = int(environment.actions()['shape'][0] / 2)

    def initial_internals(self):
        pass

    def observe(self, reward, terminal=False, parallel=0, query=None, **kwargs):
        pass

    def act(self, **kwargs):
        splitting = [(config.LAYER_NUM - 1) for _ in range(self.deviceNum * 2)]
        return splitting

    def close(self):
        pass
