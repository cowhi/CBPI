#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# import numpy as np
from experiment import Experiment
from learner_Q import LearnerQ
# import helper
import logging
_logger = logging.getLogger(__name__)


class ExperimentQ(Experiment):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

    def __init__(self, params_file):
        """ Initializes an experiment.
        """
        super(ExperimentQ, self).__init__(params_file)
        self.learner = LearnerQ(action_count=len(self.env.actions),
                                epsilon=self.params['epsilon'],
                                gamma=self.params['gamma'],
                                alpha=self.params['alpha'],
                                rng=self.rng)
        self.learner.init_Q(states=self.env.get_all_states(), how='zero')
        self.agent_name = 'agent'
        # self.set_s tatus('idle')

    def _init_task(self, *args):
        pass

    def _cleanup_task(self, *args):
        pass

    def _init_run(self):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')
        self.learner.set_epsilon(self.params['epsilon'])
        self.learner.set_epsilon(self.params['epsilon'])

    def _cleanup_run(self, *args):
        pass

    def _init_episode(self, *args):
        pass

    def _cleanup_episode(self, *args):
        if self.learner.epsilon > -1 * self.learner.epsilon_change:
            self.learner.set_epsilon(self.learner.epsilon +
                                     self.learner.epsilon_change)

    def _get_action_id(self, state, *args):
        """ Returns the action_id following a policy from the current
            library from a given state.
        """
        return self.learner.get_action(state)

    def _write_test_results(self):
        pass

    def _specific_updates(self, *args):
        pass

if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_Q.yaml')
    exp = ExperimentQ(params_file)
    exp.main()
