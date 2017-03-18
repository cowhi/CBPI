#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import collections
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
        if self.params['load_policies']:
            self.fill_library()
        self.current_library = collections.OrderedDict()
        # self.set_s tatus('idle')

    def fill_library(self):
        for learned_policy in self.params['learned_policies']:
            self.library.setdefault(learned_policy['name'], {})['goal_pos'] = \
                learned_policy['goal_pos']
            self.library.setdefault(learned_policy['name'], {})['Q'] = \
                self.learner.load_Qs(os.path.join(learned_policy['directory'],
                                                  'best_Qs.npy'))
            self.library.setdefault(learned_policy['name'], {})['importance'] \
                = 0.0

    def init_run_exp(self, task_name, run):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')

        if task_name == 'omega':
            self.learner.set_epsilon(0.2)
            self.learner.set_epsilon(0.2)
        else:
            self.learner.set_epsilon(self.params['epsilon'])
            self.learner.set_epsilon(self.params['epsilon'])
        self.run_lib_file = os.path.join(self.run_dir,
                                         'stats_policy_usage.csv')
        policy_usage_header = self.task_policies[:]
        policy_usage_header.insert(0, 'episode')
        helper.write_stats_file(self.run_lib_file,
                                policy_usage_header)
        self.tau_policy = self.params['tau_policy']
        self.evaluate_current_library(task_name, run, 0)

    def init_task_exp(self, task_name):
        # TODO: reset learner ?!
        self.init_library(task_name)
        self.build_policy_library(task_name)

    def cleanup_task_exp(self, task_name):
        helper.summarize_runs_policy_usage(self.task_dir)
        helper.plot_policy_usage_summary(self.task_dir)
        self.update_library(task_name)
        self.current_library = collections.OrderedDict()

    def cleanup_run_exp(self, path_to_dir):
        helper.plot_policy_usage(path_to_dir)

    def init_library(self, task_name):
        """ Adds the initialized policy to the library for learning the
            current task.
        """
        task_index = next(index
                          for (index, d) in enumerate(self.params['tasks'])
                          if d['name'] == task_name)
        self.library.setdefault(task_name, {})['goal_pos'] = \
            self.params['tasks'][task_index]['goal_pos']
        self.library.setdefault(task_name, {})['Q'] = self.learner.Q
        self.library.setdefault(task_name, {})['importance'] = 0.0

    def build_policy_library(self, task_name):
        """ Adds the policies from the most similar tasks from the general
            library to the library for learning the current task.

            Picks only the most similar ones and then limits the number of
            policies to a predefined number (self.params['task_library_size']).
        """
        # TODO: make more general to learn all tasks using the library
        if task_name == 'omega':
            task_index = next(index
                              for (index, d) in enumerate(self.params['tasks'])
                              if d['name'] == task_name)
            potential_policies = []
            similarity_limit = self.get_similarity_limit()
            for policy_name in self.library:
                similarity = self.get_similarity(policy_name, task_index)
                _logger.debug("%s: distance_to_%s %s <= map_diagonal %s" %
                              (policy_name, task_name,
                               str(similarity), str(similarity_limit)))
                if similarity < similarity_limit:
                    potential_policies.append([policy_name, similarity])
            potential_policies.sort(key=lambda x: x[1])
            _logger.debug("Sorted potential policies: %s" %
                          str(potential_policies))
            self.task_policies = []
            for policy_name, distance \
                    in potential_policies[0:self.params['task_library_size']]:
                self.current_library[policy_name] = \
                    self.library[policy_name]
                self.current_library[policy_name]['weight'] = 0.0
                self.current_library[policy_name]['active'] = True
                self.task_policies.append(policy_name)
        else:
            self.current_library[task_name] = self.library[task_name]
            self.current_library[task_name]['weight'] = 0.0
            self.current_library[task_name]['active'] = True
            self.task_policies = [task_name]
        _logger.info("Source policies for %s: %s" %
                     (str(task_name), str(self.task_policies)))

    def get_similarity_limit(self):
        """ Get the max distance from goal position to be relevant.
        """
        return self.params['policy_eval_factor'] * self.map_diagonal

    def get_similarity(self, policy_name, task_index):
        """ Evaluate task similarity using euclidean goal distance.
        """
        x = self.library[policy_name]['goal_pos'][0] - \
            self.params['tasks'][task_index]['goal_pos'][0]
        y = self.library[policy_name]['goal_pos'][1] - \
            self.params['tasks'][task_index]['goal_pos'][1]
        return math.sqrt(x**2 + y**2)

    def evaluate_current_library(self, task_name=None, run=None, episode=None):
        self.set_status('policy_eval', task_name, run, episode)
        divider = 0.0
        for policy_name in self.current_library:
            if self.current_library[policy_name]['active']:
                policy_results_mean = self.get_mean_test_results(
                    self.current_library[policy_name]['Q'],
                    self.current_library[policy_name]['goal_pos'])
                _logger.debug("Mean steps with policy %s: %s" %
                              (policy_name,
                               str(policy_results_mean)))
                self.current_library[policy_name]['weight'] = \
                    self.get_policy_weight(policy_results_mean)
            else:
                self.current_library[policy_name]['weight'] = 0.0
            divider += self.current_library[policy_name]['weight']
        weights = [episode]
        for policy_name in self.current_library:
            self.current_library[policy_name]['weight'] /= divider
            weights.append(self.current_library[policy_name]['weight'])
            _logger.debug("%s: weight = %s" %
                          (policy_name,
                           str(self.current_library[policy_name]['weight'])))
        helper.write_stats_file(self.run_lib_file,
                                weights)
        self.update_train_settings()
        if episode == self.params['episodes']:
            self.set_status('summarizing', task_name, run, episode)
        elif episode == 0:
            pass
        else:
            self.set_status('training', task_name, run, episode)

    def update_train_settings(self):
        """ Changes the temperature value for the softmax when calculating
            the weights for each policy.

            Also updates the current library if a policy becomes
            too unimportant.
        """
        # TODO: make this more dependent on development of learning progress
        if self.tau_policy > 1.1 * self.params['tau_policy_change']:
            # print(self.tau_policy, self.params['tau_policy_change'])
            self.tau_policy -= self.params['tau_policy_change']
        for policy_name in self.current_library:
            if self.current_library[policy_name]['weight'] < \
                    self.params['policy_importance_limit']:
                self.current_library[policy_name]['active'] = False
                _logger.debug("Deactivated policy %s" % policy_name)

    def get_policy_weight(self, policy_results_mean):
        """ Returns the weight for the used policy.
        """
        return np.exp(-1.0 * (policy_results_mean /
                      self.tau_policy) /
                      (self.params['policy_eval_episodes'] *
                       len(self.params['test_positions'])))

    def update_library(self, task_name):
        # TODO: library compare policies using score on final policy minus
        # threshold
        similar = True
        if not similar:
            # remove current task from library
            del self.library[task_name]

    def get_mean_test_results(self, policy, goal_pos):
        policy_results = []
        for i in range(0, self.params['policy_eval_episodes']):
            results_mean = self.run_policy_evals(policy, goal_pos)
            policy_results.append(results_mean)
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
        _logger.debug("Steps per test: %s" %
                      (str(self.test_steps)))
        return np.mean(self.test_steps)


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_CBPI.yaml')
    exp = ExperimentCBPI(params_file)
    exp.main()
