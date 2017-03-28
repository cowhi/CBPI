#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from experiment import Experiment
from learner_Q import LearnerQ
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

    def init_run_exp(self, name, run=None):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')
        self.learner.set_epsilon(self.params['epsilon'])
        self.learner.set_epsilon(self.params['epsilon'])

    def init_task_exp(self, name):
        pass

    def cleanup_task_exp(self, name):
        pass

    def cleanup_run_exp(self, name):
        _logger.debug("Q-values: %s" %
                      str(self.learner.Q))


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_test.yaml')
    exp = ExperimentQ(params_file)
    exp.main()
