import numpy as np


class LearnerRandom(object):
    def __init__(self, action_count=4, name='random',
                 rng=np.random.RandomState(1)):
        self.action_count = action_count
        self.name = name
        self.rng = rng
        self.last_epsilon = 0
        self.epsilon = 0
        self.epsilon_change = 0

    def get_action(self, state):
        return self.rng.random_integers(0, self.action_count - 1)

    def set_epsilon(self, epsilon):
        self.last_epsilon = self.epsilon
        self.epsilon = epsilon

    def save_Qs(self, *args):
        pass

    def update_Q(self, *args):
        pass
