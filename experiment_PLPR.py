#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# import math
import operator
import collections
# import difflib
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
        """ Initializes an experiment. """
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
        self.library = collections.OrderedDict()
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
            self.library.setdefault(learned_policy['name'], {})['W_sum'] = 0.0
            self.library.setdefault(learned_policy['name'], {})['P'] = 0.0

    def init_library(self):
        """ Adds the initialized policy to the library for learning the
            current task.
        """
        task_index = next(index
                          for (index, d) in enumerate(self.params['tasks'])
                          if d['name'] == self.current_task['name'])
        self.library.setdefault(self.current_task['name'], {})['goal_pos'] = \
            self.params['tasks'][task_index]['goal_pos']
        self.library.setdefault(self.current_task['name'],
                                {})['Q'] = self.learner.Q
        self.library.setdefault(self.current_task['name'], {})['U'] = 0
        self.library.setdefault(self.current_task['name'], {})['W_sum'] = 0.0
        self.library.setdefault(self.current_task['name'], {})['W'] = 0.0
        self.library.setdefault(self.current_task['name'], {})['P'] = 0.0

    def _init_task(self):
        self.init_library()

    def _cleanup_task(self):
        helper.summarize_runs_policy_choice(self.task_dir, 'W')
        helper.plot_policy_choice_summary(self.task_dir, 'W')
        helper.summarize_runs_policy_choice(self.task_dir, 'W_mean')
        helper.plot_policy_choice_summary(self.task_dir, 'W_mean')
        helper.summarize_runs_policy_choice(self.task_dir, 'U')
        helper.plot_policy_choice_summary(self.task_dir, 'U')
        helper.summarize_runs_policy_choice(self.task_dir, 'P')
        helper.plot_policy_choice_summary(self.task_dir, 'P')
        self.update_library()

    def update_Ps(self):
        """ Set the probability of being selected for each policy
            in the library. """
        P = collections.OrderedDict()
        for policy_name in self.library:
            try:
                P[policy_name] = np.exp(self.tau_policy *
                                        self.library[policy_name]['W'])
            except:
                P[policy_name] = 0.0
        P_sum = sum(P.values())
        for policy_name in self.library:
            try:
                self.library[policy_name]['P'] = P[policy_name] / P_sum
            except:
                self.library[policy_name]['P'] = 0.0

    def select_policy(self):
        policies = collections.OrderedDict()
        for policy_name in self.library:
            policies[policy_name] = self.library[policy_name]['P']
        sorted_policies = collections.OrderedDict(sorted(policies.items(),
                                                  key=operator.itemgetter(1),
                                                  reverse=True))
        policy_pick = self.rng.uniform(0, 1)
        upper = 0
        for policy in sorted_policies:
            lower = upper
            upper = lower + sorted_policies[policy]
            if lower <= policy_pick <= upper:
                return policy
        # just for safety in case of rounding error in last step, return the
        # most probable policy
        for policy in sorted_policies:
            return policy

    def _init_episode(self):
        self.current_W = 0.0
        self.psi = self.params['policy_reuse_probability']
        if self.status == 'training':
            self.update_Ps()
            self.current_policy = self.select_policy()
            _logger.debug('%s: current_W=%s, W=%s, U=%s, tau=%s' %
                          (str(self.current_policy),
                           str(self.current_W),
                           str(self.library[self.current_policy]['W']),
                           str(self.library[self.current_policy]['U']),
                           str(self.tau_policy)))

    def _cleanup_episode(self):
        if self.status == 'training':
            # if self.learner.epsilon > -1 * self.learner.epsilon_change:
            #    self.learner.set_epsilon(self.learner.epsilon +
            #                             self.learner.epsilon_change)
            self.current_W = \
                ((self.params['gamma'] ** self.steps_in_episode) *
                 self.reward_in_episode)
            self.library[self.current_policy]['W'] = \
                (((self.library[self.current_policy]['W'] *
                   self.library[self.current_policy]['U']) +
                  self.current_W) /
                 (self.library[self.current_policy]['U'] + 1))
            self.library[self.current_policy]['U'] += 1
            self.tau_policy += self.params['tau_policy_delta']
            _logger.debug('%s: current_W=%s, W=%s, U=%s, tau=%s' %
                          (str(self.current_policy),
                           str(self.current_W),
                           str(self.library[self.current_policy]['W']),
                           str(self.library[self.current_policy]['U']),
                           str(self.tau_policy)))
            Ws = [self.current_episode]
            W_mean = [self.current_episode]
            Us = [self.current_episode]
            Ps = [self.current_episode]
            W_sum = 0
            for policy in self.library:
                W_sum += self.library[policy]['W']
                Ws.append(self.library[policy]['W'])
                Us.append(self.library[policy]['U'])
                Ps.append(self.library[policy]['P'])
            W_mean.append(W_sum / len(self.library))
            helper.write_stats_file(self.run_lib_W_file, Ws)
            helper.write_stats_file(self.run_lib_W_mean_file, W_mean)
            helper.write_stats_file(self.run_lib_U_file, Us)
            helper.write_stats_file(self.run_lib_P_file, Ps)

    def _init_run(self):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')
        self.learner.set_epsilon(self.params['epsilon'])
        self.learner.set_epsilon(self.params['epsilon'])
        self.run_lib_W_file = os.path.join(self.run_dir,
                                           'stats_policy_W.csv')
        self.run_lib_W_mean_file = os.path.join(self.run_dir,
                                                'stats_policy_W_mean.csv')
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
        helper.write_stats_file(self.run_lib_W_mean_file,
                                ['episode', 'W_mean'])
        helper.write_stats_file(self.run_lib_U_file,
                                policy_usage_header)
        helper.write_stats_file(self.run_lib_P_file,
                                policy_usage_header)
        self.tau_policy = self.params['tau_policy']

    def _cleanup_run(self):
        helper.plot_policy_choice(self.run_dir, 'W')
        helper.plot_policy_choice(self.run_dir, 'W_mean')
        helper.plot_policy_choice(self.run_dir, 'U')
        helper.plot_policy_choice(self.run_dir, 'P')
        for policy in self.library:
            self.library[policy]['W'] = 0.0
            self.library[policy]['W_sum'] = 0.0
            self.library[policy]['U'] = 0
            self.library[policy]['P'] = 0.0

    def _get_action_id(self, state, policy_name):
        """ Returns the action_id following a policy from the current
            library from a given state.
        """
        return self.learner.get_action_id(state,
                                          self.library,
                                          policy_name,
                                          self.current_task['name'],
                                          self.status,
                                          self.psi)

    def update_library(self):
        self.set_status('library_eval')
        Ws = []
        for policy in self.library:
            # if not policy == self.current_task['name']:
            Ws.append(self.library[policy]['W'])
        if max(Ws) >= (self.params['policy_library_simimilarity'] *
                       self.library[self.current_task['name']]['W']):
            # remove current task from library
            _logger.info('Not adding %s' % str(self.current_task['name']))
            del self.library[self.current_task['name']]
        _logger.info('Library (size=%s): %s' % (str(len(self.library)),
                                                str(self.library.keys())))

    def _write_test_results(self):
        pass

    def _specific_updates(self, *args):
        if self.status == 'training':
            self.psi *= self.params['policy_reuse_probability_decay']


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_PLPR.yaml')
    exp = ExperimentPLPR(params_file)
    exp.main()
