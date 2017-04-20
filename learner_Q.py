import numpy as np
import operator
import warnings
import logging
_logger = logging.getLogger(__name__)
warnings.simplefilter("error")


class LearnerQ(object):
    def __init__(self, action_count=4, name='Q',
                 epsilon=1.0, epsilon_change=-0.0005, alpha=0.05, gamma=0.95,
                 source=None, rng=np.random.RandomState(1)):
        """ Initializes the learning strategy. """
        self.rng = rng
        self.action_count = action_count
        self.name = name
        if source is None:
            self.Q = {}
        else:
            self.Q = self.load_Qs(source)
        self.epsilon = epsilon  # probability for random action
        self.epsilon_change = epsilon_change  # change after every episode
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # discount factor for future rewards

    def set_epsilon(self, epsilon):
        """ Sets the probability for performing a random action. """
        self.last_epsilon = self.epsilon
        self.epsilon = epsilon

    def set_learningrate(self, alpha):
        """ Sets the learning rate for the Q-Update. """
        self.alpha = alpha

    def init_Q(self, states, how='zero', Q_old={}):
        """ Initializes the Q-table for all possible values according
            to a given strategy.
        """
        for state in states:
            for action in range(0, self.action_count):
                if how == 'zero':
                    self.Q[state, action] = 0.0
                elif how == 'random':
                    self.Q[state, action] = self.rng.uniform(0, 1)
                elif how == 'copy':
                    self.Q[state, action] = Q_old[state, action]
                else:
                    raise Exception("Unknown initialization method given: %s"
                                    % str(how))

    def get_Q_size(self):
        """ Gets the size of the Q-table. """
        return len(self.Q)

    def update_Q(self, state, action, reward, state_prime):
        """ Updates a single Q-value in the Q-table for a given state
            and action provided a follow-up state and a received reward.
        """
        _logger.debug("alpha:%s, gamma:%s, V':%s, Q:%s" %
                      (str(self.alpha), str(self.gamma),
                       str(self.get_V(state_prime)),
                       str(self.get_Q(state, action))))
        try:
            self.Q[state, action] += self.alpha * (reward + (self.gamma *
                                                   self.get_V(state_prime)) -
                                                   self.get_Q(state, action))
        except KeyError:
            self.Q[state, action] = self.alpha * (reward + (self.gamma *
                                                  self.get_V(state_prime)) -
                                                  self.get_Q(state, action))

    def get_Q(self, state, action):
        """ Returns a Q-value from the Q-table given a state and an action. """
        return self.Q[state, action]

    def get_V(self, state):
        """ Returns the state value for a given state. """
        Qs = [self.Q[state, action] for action in range(self.action_count)]
        return max(Qs)

    def get_action_id(self, state):
        """ Gets the action_id following the policy for the given
            task in a given state.
        """
        explore_probability = self.rng.uniform(0, 1)
        if explore_probability < self.epsilon:
            _logger.debug("### Random action (%s < %s) ###" %
                          (str(explore_probability), str(self.epsilon)))
            return self.rng.randint(0, self.action_count)
        else:
            _logger.debug("### Selecting action e-greedily "
                          "(%s > %s) ###" % (str(explore_probability),
                                             str(self.epsilon)))
            return self.greedy(self.Q, state)

    def greedy(self, policy_Qs, state, status='training'):
        """ Follows a greedy strategy selcting always the action with
            the highest Q-value. """
        Qs = [(action, policy_Qs[state, action])
              for action in range(self.action_count)]
        if status in ['library_eval']:
            if len(set(Qs)) == 1:
                return 0
        _logger.debug("Q-values in %s: %s" %
                      (str(state), str(Qs)))
        sorted_Qs = sorted(Qs, key=operator.itemgetter(1),
                           reverse=True)
        best_Q = sorted_Qs[0][1]
        max_Qs = [action for action, Q in sorted_Qs if Q == best_Q]
        return self.rng.choice(max_Qs)

    def load_Qs(self, source):
        return np.load(source).item()

    def save_Qs(self, target):
        np.save(target, self.Q)
