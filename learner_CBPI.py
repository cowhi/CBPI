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

    def get_action(self, state, library=None):
        explore_propability = self.rng.uniform(0, 1)
        if explore_propability < self.epsilon:
            return self.rng.random_integers(0, self.action_count - 1)
        else:
            library_probs = []
            for policy in library:
                Qs = []
                for action in range(0, self.action_count - 1):
                    try:
                        Qs.append((action, policy.Q[state, action]))
                    except KeyError:
                        Qs.append((action, 0.0))
                action_values = []
                for Q in Qs:
                    action_values.append(Q[1])
                action_probs = self.get_action_probs(action_values)
                weighted_action_probs = [policy.weight * i
                                         for i in action_probs]
                library_probs.append(weighted_action_probs)
            probs = [sum(i) for i in zip(*library_probs)]
            # select action using probabillities
            action_pick = self.rng.uniform(0, 1)
            for i in range(0, len(probs) - 1):
                lower = 0
                if not i == 0:
                    for j in range(0, i - 1):
                        lower += probs[j]
                upper = lower + probs[i]
                if lower <= action_pick <= upper:
                    return i

    def get_action_probs(action_values, tau=0.1):
        # calculate the softmax over the action values
        divider = 0
        for value in action_values:
            divider += np.exp(value / tau)
        action_probs = []
        for value in action_values:
            p = (np.exp(value / tau)) / divider
            action_probs.append(p)
        return action_probs

    def load_Qs(self, source):
        return np.load(source).item()

    def save_Qs(self, target):
        np.save(target, self.Q)
