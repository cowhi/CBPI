#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import operator
from learner_Q import LearnerQ
import logging
_logger = logging.getLogger(__name__)


class LearnerPLPR(LearnerQ):
    def __init__(self, action_count=4, name='PPR',
                 epsilon=1.0, epsilon_change=-0.0005, alpha=0.05, gamma=0.95,
                 source=None, rng=np.random.RandomState(1)):
        super(LearnerPLPR, self).__init__(action_count, name,
                                          epsilon, epsilon_change,
                                          alpha, gamma,
                                          source, rng)

    def get_action(self, state, library=None, policy_name=None,
                   status='training', psi=0.0):
        """ Get an action according to current policy during testing  """
        if status in ['testing', 'library_eval']:
            Qs = []
            action_values = []
            for action in range(0, self.action_count):
                try:
                    action_values.append(
                        library[policy_name]['Q'][state, action])
                except KeyError:
                    action_values.append(0.0)
            if sum(action_values) == 0.0:
                if status == 'library_eval':
                    return 0
                return self.rng.randint(0, self.action_count)
            return np.argmax(action_values)
        """ Get an action using the policy reuse strategy. """
        reuse_probability = self.rng.uniform(0, 1)
        if reuse_probability < psi:
            _logger.debug("### Reusing policy ###")
            Qs = []
            for action in range(0, self.action_count):
                try:
                    Qs.append((action,
                               library[policy_name]['Q'][state, action]))
                except KeyError:
                    pass
            _logger.debug("Q-values in %s: ) = %s" %
                          (str(state), str(Qs)))
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
                return self.rng.randint(0, self.action_count)
        else:
            explore_propability = self.rng.uniform(0, 1)
            if explore_propability < self.epsilon:
                _logger.debug("### Random action ###")
                return self.rng.randint(0, self.action_count)
            else:
                _logger.debug("### Selecting action ###")
                Qs = []
                for action in range(0, self.action_count):
                    try:
                        Qs.append((action,
                                   self.Q[state, action]))
                    except KeyError:
                        pass
                _logger.debug("Q-values in %s: ) = %s" %
                              (str(state), str(Qs)))
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
                    return self.rng.randint(0, self.action_count)
