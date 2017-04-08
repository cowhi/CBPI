#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# import math
import operator
import collections
import difflib
from experiment import Experiment
from learner_PLPR import LearnerPLPR
import helper
import numpy as np
import logging
_logger = logging.getLogger(__name__)


class ExperimentPLPR(Experiment):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

    def __init__(self, params_file):
        """ Initializes an experiment.

        """
        super(ExperimentPLPR, self).__init__(params_file)
        self.learner = LearnerPLPR(action_count=len(self.env.actions),
                                   epsilon=self.params['epsilon'],
                                   gamma=self.params['gamma'],
                                   alpha=self.params['alpha'],
                                   rng=self.rng)
        self.learner.init_Q(states=self.env.get_all_states(), how='zero')
        self.agent_name = 'agent'
        # self.map_size = self.env.get_map_size()
        # self.map_diagonal = math.sqrt(self.map_size[0]**2 +
        #                              self.map_size[1]**2)
        self.library = {}
        if self.params['load_policies']:
            self.fill_library()

    def fill_library(self):
        for learned_policy in self.params['learned_policies']:
            self.library.setdefault(learned_policy['name'], {})['goal_pos'] = \
                learned_policy['goal_pos']
            self.library.setdefault(learned_policy['name'], {})['Q'] = \
                self.learner.load_Qs(os.path.join(learned_policy['directory'],
                                                  'best_Qs.npy'))
            self.library.setdefault(learned_policy['name'], {})['U'] = 0
            self.library.setdefault(learned_policy['name'], {})['W'] = 0.0
            self.library.setdefault(learned_policy['name'], {})['P'] = 0.0

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
        self.library.setdefault(task_name, {})['U'] = 0
        self.library.setdefault(task_name, {})['W'] = 0.0
        self.library.setdefault(task_name, {})['P'] = 0.0

    def _init_task(self, task_name):
        self.init_library(task_name)

    def _cleanup_task(self, task_name):
        helper.summarize_runs_policy_choice(self.task_dir, 'W')
        helper.plot_policy_choice_summary(self.task_dir, 'W')
        helper.summarize_runs_policy_choice(self.task_dir, 'U')
        helper.plot_policy_choice_summary(self.task_dir, 'U')
        helper.summarize_runs_policy_choice(self.task_dir, 'P')
        helper.plot_policy_choice_summary(self.task_dir, 'P')
        self.update_library(task_name)

    def update_Ps(self):
        """ Set the probability of being selected for each policy
            in the library. """
        P = {}
        for policy_name in self.library:
            P[policy_name] = np.exp(self.tau_policy *
                                    self.library[policy_name]['W'])
        P_sum = sum(P.values())
        for policy_name in self.library:
            self.library[policy_name]['P'] = P[policy_name] / P_sum

    def select_policy(self):
        policies = collections.OrderedDict()
        for policy_name in self.library:
            policies[policy_name] = self.library[policy_name]['P']
        sorted_policies = collections.OrderedDict(sorted(policies.items(),
                                                  key=operator.itemgetter(1),
                                                  reverse=True))
        equally_good = []
        max_P = next(iter(sorted_policies.items()))
        for policy_name in sorted_policies:
            if sorted_policies[policy_name] == max_P[1]:
                equally_good.append(policy_name)
            else:
                break
        return self.rng.choice(equally_good)

    def _init_episode(self):
        if self.status == 'training':
            self.update_Ps()
            self.current_policy = self.select_policy()
        self.psi = self.params['policy_reuse_probability']

    def _cleanup_episode(self, task, run, episode):
        if self.learner.epsilon > -1 * self.learner.epsilon_change:
            self.learner.set_epsilon(self.learner.epsilon +
                                     self.learner.epsilon_change)
        self.library[self.current_policy]['W'] = \
            ((self.library[self.current_policy]['W'] *
             self.library[self.current_policy]['U'] +
             self.reward_in_episode) /
             (self.library[self.current_policy]['U'] + 1))
        self.library[self.current_policy]['U'] += 1
        self.tau_policy -= self.params['tau_policy_delta']

    def _init_run(self, task_name, run):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')
        self.learner.set_epsilon(self.params['epsilon'])
        self.learner.set_epsilon(self.params['epsilon'])
        self.run_lib_W_file = os.path.join(self.run_dir,
                                           'stats_policy_W.csv')
        self.run_lib_U_file = os.path.join(self.run_dir,
                                           'stats_policy_U.csv')
        self.run_lib_P_file = os.path.join(self.run_dir,
                                           'stats_policy_P.csv')
        policy_usage_header = []
        for policy_name in self.library:
            policy_usage_header.append(policy_name)
        policy_usage_header.insert(0, 'episode')
        helper.write_stats_file(self.run_lib_W_file,
                                policy_usage_header)
        helper.write_stats_file(self.run_lib_U_file,
                                policy_usage_header)
        helper.write_stats_file(self.run_lib_P_file,
                                policy_usage_header)
        self.tau_policy = self.params['tau_policy']

    def _cleanup_run(self):
        helper.plot_policy_choice(self.run_dir, 'W')
        helper.plot_policy_choice(self.run_dir, 'U')
        helper.plot_policy_choice(self.run_dir, 'P')

    def _get_action_id(self, state, policy_name):
        """ Returns the action_id following a policy from the current
            library from a given state.
        """
        return self.learner.get_action(state,
                                       self.library,
                                       policy_name,
                                       self.status,
                                       self.psi)

    def update_library(self, policy_name):
        # TODO: adapt to policy reuse
        self.set_status('library_eval', policy_name, 0, 0)
        Ws = []
        for policy in self.library:
            if not policy == policy_name:
                Ws.append(self.library[policy]['W'])
        if max(Ws) >= (self.params['policy_library_simimilarity'] *
                       self.library[policy_name]['W']):
            # remove current task from library
            _logger.info('Not adding %s' % str(policy_name))
            del self.library[policy_name]
        _logger.info('Library (size=%s): %s' % (str(len(self.library)),
                                                str(self.library.keys())))

    def _write_test_results(self):
        pass

    def _specific_updates(self, policy_name):
        Ws = [episode]
        Us = [episode]
        Ps = [episode]
        for policy in self.library:
            Ws.append(self.library[policy]['W'])
            Us.append(self.library[policy]['U'])
            Ps.append(self.library[policy]['P'])
        helper.write_stats_file(self.run_lib_W_file,
                                Ws)
        helper.write_stats_file(self.run_lib_U_file,
                                Us)
        helper.write_stats_file(self.run_lib_P_file,
                                Ps)
        self.psi *= self.params['policy_reuse_probability_decay']


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_PLPR.yaml')
    exp = ExperimentPLPR(params_file)
    exp.main()
