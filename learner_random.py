import numpy as np


class LearnerRandom(object):
    def __init__(self, action_count=4, name='random',
                 rng=np.random.RandomState(1)):
        self.action_count = action_count
        self.name = name
        self.rng = rng

    def get_action(self, state):
        return self.rng.random_integers(0, self.action_count - 1)
