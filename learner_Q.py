import numpy as np
import operator


class LearnerQ(object):
    def __init__(self, action_count=4, name='Q',
                 epsilon=1.0, epsilon_change=-0.0005, alpha=0.05, gamma=0.95,
                 source=None, rng=np.random.RandomState(1)):
        self.rng = rng
        self.action_count = action_count
        self.name = name
        if not source:
            self.Q = {}
        else:
            self.Q = self.load_Qs(source)
        self.epsilon = epsilon  # probability for random action
        self.epsilon_change = epsilon_change  # change after every episode
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # discount factor for future rewards

    def set_epsilon(self, epsilon):
        self.last_epsilon = self.epsilon
        self.epsilon = epsilon

    def set_learningrate(self, alpha):
        self.alpha = alpha

    def init_Q(self, states, how='zero', Q_old={}):
        for state in states:
            for action in range(0, self.action_count):
                if how == 'random':
                    self.Q[state, action] = self.rng.uniform(0, 1)
                if how == 'zero':
                    self.Q[state, action] = 0.0
                if how == 'copy':
                    self.Q[state, action] = Q_old[state, action]

    def get_Q_size(self):
        return len(self.Q)

    def update_Q(self, state, action, reward, state_prime):
        try:
            self.Q[state, action] += self.alpha * (reward + (self.gamma *
                                                   self.get_V(state_prime)) -
                                                   self.get_Q(state, action))
        except KeyError:
            self.Q[state, action] = self.alpha * (reward + (self.gamma *
                                                  self.get_V(state_prime)) -
                                                  self.get_Q(state, action))

    def get_Q(self, state, action):
        try:
            return self.Q[state, action]
        except KeyError:
            return 0.0

    def get_V(self, state):
        Qs = []
        for action in range(0, self.action_count):
            try:
                Qs.append(self.Q[state, action])
            except KeyError:
                Qs.append(0.0)
        return max(Qs)

    def get_action(self, state, reuse=False):
        explore_propability = self.rng.uniform(0, 1)
        if explore_propability < self.epsilon and not reuse:
            return self.rng.random_integers(0, self.action_count - 1)
        else:
            Qs = []
            for action in range(0, self.action_count):
                try:
                    Qs.append((action, self.Q[state, action]))
                except KeyError:
                    pass
            if len(Qs) > 0:
                sorted_Qs = sorted(Qs, key=operator.itemgetter(1),
                                   reverse=True)
                max_Qs = []
                max_Qs.append(sorted_Qs[0][0])
                for i in range(1, len(sorted_Qs)):
                    if sorted_Qs[i][1] == sorted_Qs[0][1]:
                        max_Qs.append(sorted_Qs[i][0])
                return self.rng.choice(max_Qs)
            else:
                return self.rng.random_integers(0, self.action_count - 1)

    def load_Qs(self, source):
        return np.load(source).item()

    def save_Qs(self, target):
        np.save(target, self.Q)
