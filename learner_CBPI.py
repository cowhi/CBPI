#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from learner_Q import LearnerQ


class LearnerCBPI(LearnerQ):
    def __init__(self, action_count=4, name='CBPI',
                 epsilon=1.0, epsilon_change=-0.0005, alpha=0.05, gamma=0.95,
                 source=None, rng=np.random.RandomState(1)):
        super(LearnerCBPI, self).__init__(action_count, name,
                                          epsilon, epsilon_change,
                                          alpha, gamma,
                                          source, rng)

    def get_action(self, state, library=None, status='training', tau=0.1):
        # TODO: FIX for testruns! and best also for eval runs!
        if status in ['policy_eval', 'testing']:
            Qs = []
            action_values = []
            for action in range(0, self.action_count):
                try:
                    # Qs.append((action, self.Q[state, action]))
                    action_values.append(self.Q[state, action])
                except KeyError:
                    # Qs.append((action, 0.0))
                    action_values.append(0.0)

            if sum(action_values) == 0.0:
                return self.rng.random_integers(0, self.action_count - 1)
            return np.argmax(action_values)
        # if training
        explore_propability = self.rng.uniform(0, 1)
        if explore_propability < self.epsilon:
            return self.rng.random_integers(0, self.action_count - 1)
        else:
            library_probs = []
            for library_name in library:
                Qs = []
                action_values = []
                for action in range(0, self.action_count):
                    try:
                        q = library[library_name]['Q'][state, action]
                    except KeyError:
                        q = 0.0
                    Qs.append((action, q))
                    action_values.append(q)
                # print(action_values)
                action_probs = self.get_action_probs(action_values, tau)
                # print(action_probs)
                weighted_action_probs = [library[library_name]['weight'] * i
                                         for i in action_probs]
                # print(weighted_action_probs)
                library_probs.append(weighted_action_probs)
            probs = [sum(i) for i in zip(*library_probs)]
            # select action using probabillities
            if sum(probs) == 0.0:
                return self.rng.random_integers(0, self.action_count - 1)
            action_pick = self.rng.uniform(0, 1)
            # print(probs, action_pick)
            for i in range(0, len(probs)):
                lower = 0
                if not i == 0:
                    for j in range(0, i):
                        lower += probs[j]
                upper = lower + probs[i]
                if lower <= action_pick <= upper:
                    return i

    def get_action_probs(self, action_values, tau):
        # calculate the softmax over the action values
        divider = 0
        for value in action_values:
            divider += np.exp(value / tau)
        action_probs = []
        for value in action_values:
            p = (np.exp(value / tau)) / divider
            action_probs.append(p)
        return action_probs
