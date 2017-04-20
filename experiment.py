#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import abc
import os
import numpy as np
import pandas as pd
import time
from ruamel import yaml
import helper
from gridworld import Gridworld
import logging
_logger = logging.getLogger(__name__)


class Experiment(object):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, params_file):
        """ Initializes an experiment.

        """
        self.params = self.get_parameter(params_file)
        self.exp_dir = self.set_exp_dir()
        _logger = self.set_logger()
        _logger.info("Initializing new experiment of type %s" %
                     str(self.params['type']))
        _logger.info("Loading parameters from %s" % str(params_file))
        _logger.info("Saving logs in %s" % str(self.exp_dir))
        # self.set_s tatus('Initializing')
        # copy parameter source
        helper.copy_file(params_file,
                         os.path.join(self.exp_dir, 'params.yaml'))
        # Mersenne Twister pseudo-random number generator
        self.rng = np.random.RandomState(self.params['random_seed'])
        # set environment
        self.env = Gridworld(grid=os.path.join(os.getcwd(),
                                               'maps',
                                               self.params['grid']),
                             max_steps=self.params['max_steps'],
                             visual=self.params['visual'],
                             rng=self.rng)
        self.current_task = 'None'
        self.current_run = 0
        self.current_episode = 0
        self.exp_steps = 0

    def get_parameter(self, file_name):
        path_to_file = os.path.join(os.getcwd(), file_name)
        with open(path_to_file, 'r') as ymlfile:
            params = yaml.load(ymlfile, Loader=yaml.Loader)
        return params

    def set_exp_dir(self):
        folder = "%s_%s_%s" % (str(time.strftime("%Y-%m-%d_%H-%M")),
                               str(self.params['type']).lower(),
                               str(self.params['grid']).lower())
        path_to_dir = os.path.join(os.getcwd(), 'logs', folder)
        return helper.create_dir(path_to_dir)

    def set_logger(self):
        # make sure no loggers are already active
        try:
            logging.root.handlers.pop()
        except IndexError:
            # if no logger exist the list will be empty and we need
            # to catch the resulting error
            pass
        if self.params['log_type'] == 'stdout':
            logging.basicConfig(level=getattr(logging,
                                              self.params['log_level'],
                                              None),
                                stream=sys.stdout,
                                format='[%(asctime)s][%(levelname)s]'
                                       '[%(module)s][%(funcName)s] '
                                       '%(message)s')
        else:
            logging.basicConfig(level=getattr(logging,
                                              self.params['log_level'],
                                              None),
                                format='[%(asctime)s][%(levelname)s]'
                                       '[%(module)s][%(funcName)s] '
                                       '%(message)s',
                                filename=os.path.join(self.exp_dir,
                                                      'experiment.log'),
                                filemode='w')
        return logging.getLogger(__name__)

    def set_status(self, status):
        self.status = status
        _logger.debug("[T:%s,R:%s,E:%s] %s" %
                      (str(self.current_task['name']), str(self.current_run),
                       str(self.current_episode), str(self.status)))

    def init_episode(self):
        self.steps_in_episode = 0
        self.reward_in_episode = 0
        self._init_episode()

    @abc.abstractmethod
    def _init_episode(self):
        pass

    def cleanup_episode(self):
        self._cleanup_episode()
        if self.status == 'training':
            if self.learner.epsilon > self.params['epsilon_limit']:
                self.learner.set_epsilon(self.learner.epsilon +
                                         self.learner.epsilon_change)

    @abc.abstractmethod
    def _cleanup_episode(self):
        pass

    def init_run(self):
        _logger.info("..... Starting run %s" % str(self.current_run))
        run_dir = os.path.join(self.task_dir, 'run_' + str(self.current_run))
        self.run_dir = helper.create_dir(run_dir)
        # Create run stats file: run_stats.csv
        self.run_stats_file = os.path.join(self.run_dir,
                                           'stats_run.csv')
        self.run_steps = 0
        helper.write_stats_file(self.run_stats_file,
                                'episode',
                                'steps_total', 'steps_mean',
                                'reward_total', 'reward_mean',
                                'epsilon', 'step_count')
        self._init_run()

    @abc.abstractmethod
    def _init_run(self):
        pass

    def cleanup_run(self):
        self.save_best_episode()
        helper.delete_dirs(self.run_dir)
        helper.plot_run(self.run_dir)
        self._cleanup_run()
        _logger.info("..... Finished run %s" % str(self.current_run))

    @abc.abstractmethod
    def _cleanup_run(self):
        pass

    def init_task(self):
        _logger.info("##### Starting task %s" %
                     str(self.current_task['name']))
        task_dir = os.path.join(self.exp_dir, 'task_' +
                                self.current_task['name'])
        self.task_dir = helper.create_dir(task_dir)
        self._init_task()

    @abc.abstractmethod
    def _init_task(self):
        pass

    def cleanup_task(self):
        helper.plot_runs(self.task_dir)
        helper.summarize_runs_results(self.task_dir)
        helper.plot_task(self.task_dir)
        self.save_best_run()
        self._cleanup_task()
        # self.set_s tatus('idle')
        _logger.info("##### Finished task %s" % str(self.current_task['name']))

    @abc.abstractmethod
    def _cleanup_task(self):
        pass

    # def evaluate_current_lib rary(self):
    #    pass

    def get_action_id(self, state, policy_name):
        return self._get_action_id(state, policy_name)

    @abc.abstractmethod
    def _get_action_id(self):
        pass

    @abc.abstractmethod
    def _specific_updates(self):
        pass

    def write_test_results(self):
        helper.write_stats_file(self.run_stats_file,
                                self.current_episode,
                                sum(self.test_steps),
                                np.mean(self.test_steps),
                                sum(self.test_rewards),
                                np.mean(self.test_rewards),
                                float("{0:.5f}"
                                      .format(self.learner.last_epsilon)),
                                self.run_steps)
        self._write_test_results()

    @abc.abstractmethod
    def _write_test_results(self):
        pass

    def run_tests(self):
        self.learner.set_epsilon(0.0)
        self.episode_dir = os.path.join(self.run_dir,
                                        'episode_' + str(self.current_episode))
        self.episode_dir = helper.create_dir(self.episode_dir)
        self.test_steps = []
        self.test_rewards = []
        for test_pos in self.params['test_positions']:
            self.init_episode()
            self.run_episode(test_pos,
                             tuple(self.current_task['goal_pos']),
                             self.current_task['name'])
            self.test_steps.append(self.steps_in_episode)
            self.test_rewards.append(self.reward_in_episode)
        self.write_test_results()
        self.learner.save_Qs(os.path.join(self.episode_dir,
                                          'Qs.npy'))
        # Make video from random position
        if self.params['visual']:
            self.set_status('recording')
            self.init_episode()
            self.run_episode(
                self.env.get_random_state(
                    tuple(self.current_task['goal_pos'])),
                tuple(self.current_task['goal_pos']),
                self.current_task['name'])
        self.learner.set_epsilon(self.learner.last_epsilon)

    def run_episode(self, agent_pos, goal_pos, policy_name=None):
        """
            Function to run a single episode.
        """
        if self.status == 'training':
            _logger.debug("Start episode")
        self.env.reset_env()
        self.env.add_agent(agent_pos, self.agent_name)
        self.env.add_goal(goal_pos)
        if self.status == 'recording':
            self.env.draw_frame()
            self.env.save_current_frame(self.episode_dir)
        state = self.env.get_current_state(self.agent_name)
        action_id = self.get_action_id(state, policy_name)
        reward = self.env.step(self.env.actions[action_id],
                               self.agent_name)
        state_prime = self.env.get_current_state(self.agent_name)
        if self.status in ['training', 'policy_eval']:
            self.run_steps += 1
        if self.status == 'recording':
            self.env.draw_frame()
            self.env.save_current_frame(self.episode_dir)
        self.steps_in_episode += 1
        self.reward_in_episode += reward
        if self.status == 'training' and not self.env.episode_ended:
            self.learner.update_Q(state[0:2], action_id,
                                  reward, state_prime[0:2])
        self._specific_updates(policy_name)
        while not self.env.episode_ended:
            state = state_prime
            action_id = self.get_action_id(state, policy_name)
            reward = self.env.step(self.env.actions[action_id],
                                   self.agent_name)
            state_prime = self.env.get_current_state(self.agent_name)
            if self.status in ['training', 'policy_eval']:
                self.run_steps += 1
            if self.status == 'recording':
                self.env.draw_frame()
                self.env.save_current_frame(self.episode_dir)
            # if self.status in ['testing', 'policy_eval']:
            self.steps_in_episode += 1
            self.reward_in_episode += reward
            if self.status == 'training':
                self.learner.update_Q(state[0:2], action_id,
                                      reward, state_prime[0:2])
            if self.env.step_count >= self.env.max_steps:
                self.env.episode_ended = True
            self._specific_updates(policy_name)
        if self.status == 'training':
            _logger.debug("End episode")
        if self.env.visual and self.status == 'recording':
            self.env.make_video(self.episode_dir)

    def save_best_episode(self):
        df = pd.read_csv(os.path.join(self.run_dir, 'stats_run.csv'))
        least_steps_row = df.ix[df['steps_mean'].idxmin()]
        run_best_file = os.path.join(self.run_dir, 'stats_run_best.csv')
        headers = ['run']
        content = [int(self.current_run)]
        for column in df:
            headers.append(str(column))
            content.append(least_steps_row[column])
        helper.write_stats_file(run_best_file, headers)
        helper.write_stats_file(run_best_file, content)
        helper.copy_file(os.path.join(self.run_dir,
                                      'episode_' +
                                      str(int(least_steps_row['episode'])),
                                      'Qs.npy'),
                         os.path.join(self.run_dir,
                                      'best_Qs.npy'))

    def save_best_run(self):
        # Save best Q-table for current task
        df = pd.read_csv(os.path.join(self.task_dir,
                                      'run_' + str(1),
                                      'stats_run_best.csv'))
        for i in range(2, self.params['runs']):
            df.append(pd.read_csv(os.path.join(self.task_dir,
                                               'run_' + str(i),
                                               'stats_run_best.csv')),
                      ignore_index=True)
        least_steps_row = df.ix[df['steps_mean'].idxmin()]
        task_best_file = os.path.join(self.task_dir, 'stats_task_best.csv')
        headers = ['task']
        content = [str(self.current_task['name'])]
        for column in df:
            headers.append(str(column))
            content.append(least_steps_row[column])
        helper.write_stats_file(task_best_file, headers)
        helper.write_stats_file(task_best_file, content)
        helper.copy_file(os.path.join(self.task_dir,
                                      'run_' +
                                      str(int(least_steps_row['run'])),
                                      'best_Qs.npy'),
                         os.path.join(self.task_dir,
                                      'best_Qs.npy'))

    def main(self):
        for task in self.params['tasks']:
            self.current_task = task
            self.init_task()
            for run in range(1, self.params['runs'] + 1):
                self.current_run = run
                self.current_episode = 0
                self.current_policy = self.current_task['name']
                self.init_run()
                self.set_status('testing')
                self.run_tests()
                self.set_status('training')
                for episode in range(1, self.params['episodes'] + 1):
                    self.current_episode = episode
                    self.init_episode()
                    self.run_episode(
                        self.env.get_random_state(
                            tuple(self.current_task['goal_pos'])),
                        tuple(self.current_task['goal_pos']),
                        self.current_policy)
                    self.cleanup_episode()
                    if episode % self.params['test_interval'] == 0:
                        self.set_status('testing')
                        self.run_tests()
                        self.set_status('training')
                self.cleanup_run()
            self.cleanup_task()
        _logger.info("Done")


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_Q.yaml')
    exp = Experiment(params_file)
    exp.main()
