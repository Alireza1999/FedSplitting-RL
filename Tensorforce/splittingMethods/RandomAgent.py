import numpy as np


class RandomAgent:

    def __init__(self, environment):
        self.deviceNum = int(environment.actions()['shape'][0] / 2)

    def initial_internals(self):
        pass

    def observe(self, reward, terminal=False, parallel=0, query=None, **kwargs):
        pass

    def act(self, **kwargs):
        splitting = np.random.uniform(low=0.0, high=1.0, size=(self.deviceNum * 2))
        return splitting

    def close(self):
        pass
