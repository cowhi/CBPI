#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from experiment import Experiment
from learner_random import LearnerRandom
import logging
_logger = logging.getLogger(__name__)


class ExperimentRandom(Experiment):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

    def __init__(self, params_file):
        """ Initializes an experiment.

        """
        super(ExperimentRandom, self).__init__(params_file)
        self.learner = LearnerRandom(action_count=len(self.env.actions),
                                     rng=self.rng)
        self.agent_name = 'agent'
        self.set_status('idle')

    def init_task_exp(self):
        pass

    def cleanup_task_exp(self):
        pass


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_random.yaml')
    exp = ExperimentRandom(params_file)
    exp.main()
