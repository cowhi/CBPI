#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import collections
import difflib
from experiment import Experiment
from learner_PPR import LearnerPPR
import helper
import numpy as np
import logging
_logger = logging.getLogger(__name__)


class ExperimentPPR(Experiment):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

    def __init__(self, params_file):
        """ Initializes an experiment.

        """
        super(ExperimentPPR, self).__init__(params_file)
        self.learner = LearnerPPR(action_count=len(self.env.actions),
                                  epsilon=self.params['epsilon'],
                                  gamma=self.params['gamma'],
                                  alpha=self.params['alpha'],
                                  rng=self.rng)
        self.learner.init_Q(states=self.env.get_all_states(), how='zero')
        self.agent_name = 'agent'
        # self.map_size = self.env.get_map_size()
        # self.map_diagonal = math.sqrt(self.map_size[0]**2 +
        #                               self.map_size[1]**2)
        self.library = {}
        if self.params['load_policies']:
            self.fill_library()
        self.W = 0.0
        # self.current_library = collections.OrderedDict()

    def fill_library(self):
        for learned_policy in self.params['learned_policies']:
            self.library.setdefault(learned_policy['name'], {})['goal_pos'] = \
                learned_policy['goal_pos']
            self.library.setdefault(learned_policy['name'], {})['Q'] = \
                self.learner.load_Qs(os.path.join(learned_policy['directory'],
                                                  'best_Qs.npy'))
            self.library.setdefault(learned_policy['name'], {})['U'] \
                = 0
            self.library.setdefault(learned_policy['name'], {})['W'] \
                = 0.0
            self.library.setdefault(learned_policy['name'], {})['P'] \
                = 0.0

    def init_run_exp(self, task_name, run):
        self.learner.init_Q(states=self.env.get_all_states(),
                            how='zero')
        # if task_name == 'omega':
        #    self.learner.set_epsilon(0.2)
        #    self.learner.set_epsilon(0.2)
        # else:
        self.learner.set_epsilon(self.params['epsilon'])
        self.learner.set_epsilon(self.params['epsilon'])
        self.run_lib_probs_file = os.path.join(self.run_dir,
                                               'stats_policy_probs.csv')
        self.run_lib_absolute_file = os.path.join(self.run_dir,
                                                  'stats_policy_absolute.csv')
        policy_usage_header = self.task_policies[:]
        policy_usage_header.insert(0, 'episode')
        helper.write_stats_file(self.run_lib_probs_file,
                                policy_usage_header)
        helper.write_stats_file(self.run_lib_absolute_file,
                                policy_usage_header)
        self.W = 0.0
        # self.tau_policy = self.params['tau_policy']
        # for policy_name in self.library:
        #     self.current_library[policy_name]['active'] = True
        # self.active_policies = len(self.task_policies)
        self.evaluate_current_library(task_name, run, 0)

    def init_task_exp(self, task_name):
        # TODO: reset learner ?!
        self.init_library(task_name)
        # self.build_policy_library(task_name)

    def cleanup_task_exp(self, task_name):
        helper.summarize_runs_policy_choice(self.task_dir, 'probs')
        helper.plot_policy_choice_summary(self.task_dir, 'probs')
        helper.summarize_runs_policy_choice(self.task_dir, 'absolute')
        helper.plot_policy_choice_summary(self.task_dir, 'absolute')
        self.update_library(task_name)
        self.current_library = collections.OrderedDict()

    def cleanup_run_exp(self, path_to_dir):
        helper.plot_policy_choice(path_to_dir, 'probs')
        helper.plot_policy_choice(path_to_dir, 'absolute')

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

    '''
    def build_policy_library(self, task_name):
        """ Adds the policies from the most similar tasks from the general
            library to the library for learning the current task.

            Picks only the most similar ones and then limits the number of
            policies to a predefined number (self.params['task_library_size']).
        """
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
            self.current_library[policy_name]['confidence'] = \
                self.params['policy_eval_confidence']
            self.task_policies.append(policy_name)
        _logger.info("Source policies for %s: %s" %
                     (str(task_name), str(self.task_policies)))
    '''

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
        # TODO: Implement PPR policy evaluation
        if self.W_max < self.params['policy_importance_limit'] * self.W:
            # add to library!
            # or better not remove from library?!
            pass



        removed_policy = True
        while removed_policy:
            removed_policy = False
            divider = 0.0
            absolute = [episode]
            for policy_name in self.current_library:
                if self.current_library[policy_name]['active']:
                    if self.active_policies > 1:
                        policy_results_mean = self.get_mean_test_results(
                            policy_name,
                            task_name,
                            episode)
                        self.current_library[policy_name]['weight'] = \
                            self.get_policy_weight(policy_results_mean)
                        absolute.append(policy_results_mean)
                    else:
                        absolute.append(0.0)
                else:
                    self.current_library[policy_name]['weight'] = 0.0
                    absolute.append(0.0)
                divider += self.current_library[policy_name]['weight']
            weights = [episode]
            for policy_name in self.current_library:
                if self.current_library[policy_name]['active']:
                    if self.active_policies > 1:
                        self.current_library[policy_name]['weight'] /= divider
                        if self.current_library[policy_name]['weight'] < \
                                self.params['policy_importance_limit'] and not\
                                policy_name == task_name:
                            self.current_library[policy_name]['active'] = False
                            self.active_policies -= 1
                            _logger.debug("Episode %s: Deactivated policy %s" %
                                          (str(episode), policy_name))
                            removed_policy = True
                            break
                    else:
                        self.current_library[policy_name]['weight'] = 1.0
                weights.append(self.current_library[policy_name]['weight'])
                _logger.debug(
                    "%s: weight = %s" %
                    (policy_name,
                     str(self.current_library[policy_name]['weight'])))
        helper.write_stats_file(self.run_lib_absolute_file,
                                absolute)
        helper.write_stats_file(self.run_lib_probs_file,
                                weights)
        # self.update_train_settings()


        if episode == self.params['episodes']:
            self.set_status('summarizing', task_name, run, episode)
        elif episode == 0:
            pass
        else:
            self.set_status('training', task_name, run, episode)

    '''
    def update_train_settings(self):
        """ Changes the temperature value for the softmax when calculating
            the weights for each policy.

            Also updates the current library if a policy becomes
            too unimportant.
        """
        # TODO FUTURE: make this more dependent on development
        # of learning progress
        if self.tau_policy > 1.1 * self.params['tau_policy_change']:
            self.tau_policy -= self.params['tau_policy_change']
    '''

    def get_policy_weight(self, policy_results_mean):
        """ Returns the weight for the used policy.
        """
        return np.exp(-1.0 * (policy_results_mean /
                      self.tau_policy) /
                      len(self.params['test_positions']))

    def update_library(self, task_name):
        self.set_status('library_eval', task_name, 0, 0)
        self.library[task_name]['Q'] = \
            self.learner.load_Qs(os.path.join(self.task_dir,
                                              'best_Qs.npy'))
        if len(self.task_policies) > 1 and \
                not self.add_to_library(task_name):
            # remove current task from library
            _logger.info('Not adding %s' % str(task_name))
            del self.library[task_name]
        # policies = []
        # for policy_name in self.library:
        #    policies.append(policy_name)
        _logger.info('Library (size=%s): %s' % (str(len(self.library)),
                                                str(self.library.keys())))

    def add_to_library(self, task_name):
        # TODO: library compare policies using score on final policy minus
        # threshold
        eval_lib = {}
        for policy_name in self.current_library:
            eval_lib[policy_name] = []
        for i in range(0, self.params['policy_eval_states']):
            # 1) select random state
            random_state = self.env.get_random_state(
                tuple(self.library[task_name]['goal_pos']))
            # 2) get actions for each policy in each random state
            for policy_name in self.current_library:
                action_id = self.learner.get_action(random_state,
                                                    self.current_library,
                                                    policy_name,
                                                    self.status,
                                                    self.params['tau_action'])
                eval_lib[policy_name].append(action_id)
                # TODO: count to total steps
        # 3) compare parity with current_library
        similarities = []
        for policy_name in self.current_library:
            if not policy_name == task_name:
                # workaround to avoid errors with difflib
                count_steps = 0.0
                ratio = 0.0
                for i in range(0, self.params['policy_eval_states'] + 1, 100):
                    sm = difflib.SequenceMatcher(
                            None,
                            eval_lib[task_name][i:i+100],
                            eval_lib[policy_name][i:i+100])
                    count_steps += 1.0
                    ratio += sm.ratio()
                ratio = ratio / count_steps
                _logger.info('Overlap %s - %s: %s' %
                             (str(task_name), str(policy_name),
                              str(ratio)))
                similarities.append(ratio)
                # if sm.ratio() > 0.9:
                #    return False
        if max(similarities) > self.params['policy_similarity_limit']:
            return False
        # 4) return true if very different
        return True

    def get_mean_test_results(self, policy_name, task_name, episode):
        policy_results = []
        for i in range(0, self.params['policy_eval_episodes']):
            results_mean = \
                self.run_policy_evals(policy_name, task_name)
            policy_results.append(results_mean)
        return np.mean(policy_results)

    def run_policy_evals(self, policy_name, task_name):
        self.test_steps = []
        self.test_rewards = []
        _logger.debug('Eval %s with: %s' % (str(task_name), str(policy_name)))
        for test_pos in self.params['test_positions']:
            self.init_episode()
            self.current_library[policy_name]['confidence'] = \
                self.params['policy_eval_confidence']
            self.run_episode(
                test_pos,
                tuple(self.current_library[task_name]['goal_pos']),
                policy_name)
            self.test_steps.append(self.steps_in_episode)
            self.test_rewards.append(self.reward_in_episode)
        _logger.debug("Steps per test: %s" %
                      (str(self.test_steps)))
        return np.mean(self.test_steps)


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_PPR.yaml')
    exp = ExperimentPPR(params_file)
    exp.main()
