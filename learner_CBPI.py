#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from learner_Q import LearnerQ
import logging
_logger = logging.getLogger(__name__)


class LearnerCBPI(LearnerQ):
    def __init__(self, action_count=4, name='CBPI',
                 epsilon=1.0, epsilon_change=-0.0005, alpha=0.05, gamma=0.95,
                 source=None, rng=np.random.RandomState(1)):
        super(LearnerCBPI, self).__init__(action_count, name,
                                          epsilon, epsilon_change,
                                          alpha, gamma,
                                          source, rng)

    def get_action(self, state, library=None, eval_policy=None,
                   status='training', tau=0.1):
        if status in ['testing', 'library_eval']:
            Qs = []
            action_values = []
            for action in range(0, self.action_count):
                try:
                    action_values.append(
                        library[eval_policy]['Q'][state, action])
                except KeyError:
                    action_values.append(0.0)
            if sum(action_values) == 0.0:
                if status == 'library_eval':
                    return 0
                return self.rng.randint(0, self.action_count)
            return np.argmax(action_values)
        if status in ['policy_eval']:
            doubt = self.rng.uniform(0, 1)
            if library[eval_policy]['confidence'] > doubt:
                Qs = []
                action_values = []
                for action in range(0, self.action_count):
                    try:
                        action_values.append(
                            library[eval_policy]['Q'][state, action])
                    except KeyError:
                        action_values.append(0.0)
                if sum(action_values) == 0.0:
                    return self.rng.randint(0, self.action_count)
                return np.argmax(action_values)
            else:
                return self.rng.randint(0, self.action_count)
        # if training
        explore_propability = self.rng.uniform(0, 1)
        if explore_propability < self.epsilon:
            _logger.debug("### Random action ###")
            return self.rng.randint(0, self.action_count)
        else:
            _logger.debug("### Selecting action ###")
            library_probs = []
            for library_name in library:
                if library[library_name]['active']:
                    Qs = []
                    action_values = []
                    for action in range(0, self.action_count):
                        try:
                            q = library[library_name]['Q'][state, action]
                        except KeyError:
                            q = 0.0
                        Qs.append((action, q))
                        action_values.append(q)
                    _logger.debug("%s: action_values = %s" %
                                  (str(library_name), str(action_values)))
                    transformed_action_values = \
                        self.transform_action_values(action_values)
                    _logger.debug("%s: transformed_action_values = %s" %
                                  (str(library_name),
                                   str(transformed_action_values)))
                    action_probs = \
                        self.get_action_probs(transformed_action_values,
                                              tau)
                    _logger.debug("%s: action_probs = %s" %
                                  (str(library_name), str(action_probs)))
                    weighted_action_probs = \
                        [library[library_name]['weight'] * i
                         for i in action_probs]
                    _logger.debug("%s: weighted_action_probs = %s" %
                                  (str(library_name),
                                   str(weighted_action_probs)))
                else:
                    weighted_action_probs = \
                        [0.0 for action in range(0, self.action_count)]
                    _logger.debug("%s: weighted_action_probs = %s" %
                                  (str(library_name),
                                   str(weighted_action_probs)))
                library_probs.append(weighted_action_probs)
            _logger.debug("library_probs = %s" % str(library_probs))
            probs = [sum(i) for i in zip(*library_probs)]
            _logger.debug("***** final_probs = %s" % str(probs))
            # select action using probabillities
            if sum(probs) == 0.0:
                return self.rng.randint(0, self.action_count)
            action_pick = self.rng.uniform(0, 1)
            _logger.debug("action_pick = %s" % str(action_pick))
            for i in range(0, len(probs)):
                lower = 0
                if not i == 0:
                    for j in range(0, i):
                        lower += probs[j]
                upper = lower + probs[i]
                if lower <= action_pick <= upper:
                    return i
            # just for safety in case of rounding error in last step
            return self.rng.randint(0, self.action_count)

    def get_action_probs(self, action_values, tau):
        """ Calculate the softmax over the given action values.
        """
        divider = 0
        for value in action_values:
            divider += np.exp(value / tau)
        action_probs = []
        for value in action_values:
            p = (np.exp(value / tau)) / divider
            action_probs.append(p)
        return action_probs

    def transform_action_values(self, action_values):
        """ Calculate a transformation that makes very small values more
            usable for the softmax calculation.
        """
        # get largest number
        largest = max(action_values)
        # transform all values
        transformed_action_values = []
        for value in action_values:
            if not value == 0.0:
                transformed_action_values.append(
                    value / 10 ** (math.floor(math.log(largest, 10)) + 1))
            else:
                transformed_action_values.append(value)
        return transformed_action_values
