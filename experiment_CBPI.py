#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
from experiment import Experiment
from learner_CBPI import LearnerCBPI
import helper
import numpy as np
import logging
_logger = logging.getLogger(__name__)


class ExperimentCBPI(Experiment):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

    def __init__(self, params_file):
        """ Initializes an experiment.

        """
        super(ExperimentCBPI, self).__init__(params_file)
        self.learner = LearnerCBPI(action_count=len(self.env.actions),
                                   epsilon=self.params['epsilon'],
                                   gamma=self.params['gamma'],
                                   alpha=self.params['alpha'],
                                   rng=self.rng)
        self.learner.init_Q(states=self.env.get_all_states(), how='zero')
        self.agent_name = 'agent'
        self.map_size = self.env.get_map_size()
        self.map_diagonal = math.sqrt(self.map_size[0]**2 +
                                      self.map_size[1]**2)
        self.library = {}
        self.task_library = {}
        self.current_library = {}
        self.set_status('idle')

    def init_run_exp(self, task_name):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')

        if task_name == 'omega':
            self.learner.set_epsilon(0.2)
            self.learner.set_epsilon(0.2)
        else:
            self.learner.set_epsilon(self.params['epsilon'])
            self.learner.set_epsilon(self.params['epsilon'])
        self.run_lib_file = os.path.join(self.run_dir,
                                         'stats_libs.csv')
        temp = self.task_policies
        temp.insert(0, 'episode')
        helper.write_stats_file(self.run_lib_file,
                                temp)
        self.evaluate_current_library(0)

    def init_task_exp(self, task_name):
        # TODO: reset learner
        self.init_library(task_name)
        self.build_task_library(task_name)

    def cleanup_task_exp(self, name):
        self.update_library(name)
        self.current_library = {}
        # TODO: get policy probability distribution from best run and plot!

    def cleanup_run_exp(self, path_to_dir):
        helper.plot_stats_libs(path_to_dir)

    def init_library(self, task_name):
        task_index = next(index
                          for (index, d) in enumerate(self.params['tasks'])
                          if d['name'] == task_name)
        self.library.setdefault(task_name, {})['goal_pos'] = \
            self.params['tasks'][task_index]['goal_pos']
        self.library.setdefault(task_name, {})['Q'] = self.learner.Q
        self.library.setdefault(task_name, {})['importance'] = 0.0

    def build_task_library(self, task_name):
        # evaluate task similarity using euclidean goal distance
        # TODO: turn results around so closest is most similar
        if task_name == 'omega':
            task_index = next(index
                              for (index, d) in enumerate(self.params['tasks'])
                              if d['name'] == task_name)
            self.task_policies = []
            for library_name in self.library:
                x = self.library[library_name]['goal_pos'][0] - \
                    self.params['tasks'][task_index]['goal_pos'][0]
                y = self.library[library_name]['goal_pos'][1] - \
                    self.params['tasks'][task_index]['goal_pos'][1]
                distance = math.sqrt(x**2 + y**2)
                max_distance = \
                    self.params['policy_eval_factor'] * self.map_diagonal
                _logger.debug("%s: distance_to_%s %s <= map_diagonal %s" %
                              (library_name, task_name,
                               str(distance), str(max_distance)))
                if distance < max_distance:
                    self.current_library[library_name] = \
                        self.library[library_name]
                    self.current_library[library_name]['weight'] = 0.0
                    self.task_policies.append(library_name)
        else:
            self.current_library[task_name] = self.library[task_name]
            self.current_library[task_name]['weight'] = 0.0
            self.task_policies = [task_name]
        self.task_library = self.current_library
        _logger.info("Start training %s with %s" %
                     (str(task_name), str(self.task_policies)))

    def evaluate_current_library(self, episode):
        self.set_status('policy_eval')
        divider = 0.0
        for library_name in self.current_library:
            policy_results_mean = self.get_mean(
                self.current_library[library_name]['Q'],
                self.current_library[library_name]['goal_pos'])
            _logger.debug("%s: results_mean = %s" %
                          (library_name,
                           str(policy_results_mean)))
            self.current_library[library_name]['weight'] = \
                np.exp(-1.0 * (policy_results_mean /
                       self.params['tau_policy']) /
                       (self.params['policy_eval_episodes'] *
                       len(self.params['test_positions'])))
            divider += self.current_library[library_name]['weight']
        weights = [episode]
        for library_name in self.current_library:
            self.current_library[library_name]['weight'] /= divider
            # self.current_library[library_name]['weight'] = \
            #    1 - self.current_library[library_name]['weight']
            # TODO: Be careful! Not ordered!!!
            weights.append(self.current_library[library_name]['weight'])
            _logger.debug("%s: weight = %s" %
                          (library_name,
                           str(self.current_library[library_name]['weight'])))
            # TODO: save weights for plotting
            # 'episode', 'name1', 'name2', ...
            # 0, 0.2, 0.3,
        helper.write_stats_file(self.run_lib_file,
                                weights)
        self.set_status('training')

    def update_library(self, task_name):
        # TODO: library compare policies using score on final policy minus
        # threshold
        similar = True
        if not similar:
            # remove current task from library
            del self.library[task_name]

    def get_mean(self, policy, goal_pos):
        policy_results = []
        for i in range(0, self.params['policy_eval_episodes']):
            reward_mean = self.run_policy_evals(policy, goal_pos)
            policy_results.append(reward_mean)
        return np.mean(policy_results)

    def run_policy_evals(self, policy, goal_pos):
        self.test_steps = []
        self.test_rewards = []
        for test_pos in self.params['test_positions']:
            self.init_episode()
            self.run_episode(test_pos,
                             tuple(goal_pos),
                             policy)
            self.test_steps.append(self.steps_in_episode)
            self.test_rewards.append(self.reward_in_episode)
        _logger.debug("total_test_steps = %s" %
                      (str(self.test_steps)))
        return np.mean(self.test_steps)


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_CBPI.yaml')
    exp = ExperimentCBPI(params_file)
    exp.main()
