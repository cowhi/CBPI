#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
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

    def get_action(self, state, library=None, policy_name=None, task_name=None,
                   status='training', psi=0.0):
        """ Get an action according to current policy during testing and when
           the chosen policy is the policy for the current task. """
        if status in ['testing'] or \
                policy_name == task_name:
            _logger.debug("### Selecting policy %s greedily ###" %
                          str(policy_name))
            return self.greedy(self.Q, state)
        """ Get an action using the policy reuse strategy. """
        reuse_probability = self.rng.uniform(0, 1)
        if reuse_probability < psi:
            _logger.debug("### Reusing policy %s greedily (%s < %s)###" %
                          (str(policy_name), str(reuse_probability), str(psi)))
            return self.greedy(library[policy_name]['Q'], state)
        else:
            explore_propability = self.rng.uniform(0, 1)
            epsilon = 1.0 - psi
            if explore_propability < epsilon:
                _logger.debug("### Random action (%s < %s) ###" %
                              (str(explore_propability), str(epsilon)))
                return self.rng.randint(0, self.action_count)
            else:
                _logger.debug("### Selecting policy %s e-greedily "
                              "(%s > %s) ###" % (str(task_name),
                                                 str(explore_propability),
                                                 str(epsilon)))
                return self.greedy(self.Q, state)
