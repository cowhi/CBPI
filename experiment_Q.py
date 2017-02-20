#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import time
from ruamel import yaml
import helper
from gridworld import Gridworld
from learner_Q import LearnerQ
import logging
_logger = logging.getLogger(__name__)


class Experiment(object):
    """ This is the base class for all experiment implementations.

    The experiment organizes all objects and directs the training in a given
    scenario.

    """

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
        self.learner = LearnerQ(action_count=len(self.env.actions),
                                epsilon=self.params['epsilon'],
                                gamma=self.params['gamma'],
                                alpha=self.params['alpha'],
                                rng=self.rng)
        self.learner.init_Q(states=self.env.get_all_states(), how='zero')
        self.agent_name = 'agent'
        self.status = 'idle'
        _logger.debug("Current status is %s" % str(self.status))

    def get_parameter(self, file_name):
        path_to_file = os.path.join(os.getcwd(), file_name)
        with open(path_to_file, 'r') as ymlfile:
            params = yaml.load(ymlfile, Loader=yaml.Loader)
        # print(params)
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
            logging.basicConfig(level=getattr(logging, 'INFO', None),
                                stream=sys.stdout,
                                format='[%(asctime)s][%(levelname)s]'
                                       '[%(module)s][%(funcName)s] '
                                       '%(message)s')
        else:
            logging.basicConfig(level=getattr(logging, 'INFO', None),
                                format='[%(asctime)s][%(levelname)s]'
                                       '[%(module)s][%(funcName)s] '
                                       '%(message)s',
                                filename=os.path.join(self.exp_dir,
                                                      'experiment.log'),
                                filemode='w')
        return logging.getLogger(__name__)

    def run_episode(self, agent_pos, goal_pos):
        """
            Function to run a single episode
        """
        self.env.reset_env()
        self.env.add_agent(agent_pos, self.agent_name)
        self.env.add_goal(goal_pos)
        if self.status == 'recording':
            self.env.draw_frame()
            self.env.save_current_frame(self.episode_dir)
        state = self.env.get_current_state(self.agent_name)
        action_id = self.learner.get_action(state)
        reward = self.env.step(self.env.actions[action_id],
                               self.agent_name)
        state_prime = self.env.get_current_state(self.agent_name)
        if self.status == 'testing':
            self.steps_in_episode += 1
            self.reward_in_episode += reward
        if self.status == 'training':
            self.learner.update_Q(state[0:2], action_id,
                                  reward, state_prime[0:2])
        if self.status == 'recording':
            self.env.draw_frame()
            self.env.save_current_frame(self.episode_dir)
        while not self.env.episode_ended:
            state = state_prime
            action_id = self.learner.get_action(state)
            reward = self.env.step(self.env.actions[action_id],
                                   self.agent_name)
            state_prime = self.env.get_current_state(self.agent_name)
            if self.status == 'testing':
                self.steps_in_episode += 1
                self.reward_in_episode += reward
            if self.status == 'training':
                self.learner.update_Q(state[0:2], action_id,
                                      reward, state_prime[0:2])
            if self.status == 'recording':
                self.env.draw_frame()
                self.env.save_current_frame(self.episode_dir)
            if self.env.step_count >= self.env.max_steps:
                self.env.episode_ended = True
        if self.env.visual and self.status == 'recording':
            self.env.make_video(self.episode_dir)

    def init_episode(self):
        self.steps_in_episode = 0
        self.reward_in_episode = 0

    def run(self):
        for task in self.params['tasks']:
            # create task directory
            task_dir = os.path.join(self.exp_dir, 'task_' + task['name'])
            self.task_dir = helper.create_dir(task_dir)
            # Create task stats file: task_stats.csv
            # self.task_stats_file = os.path.join(self.task_dir,
            #                                    'stats_task.csv')
            # helper.write_stats_file(self.task_stats_file,
            #                        'episode',
            #                       'steps_mean', 'steps_lower', 'steps_upper',
            #                        'reward_mean', 'reward_lower',
            #                        'reward_upper')
            _logger.info("Starting task %s" % str(task['name']))
            for run in range(1, self.params['runs'] + 1):
                run_dir = os.path.join(self.task_dir, 'run_' + str(run))
                self.run_dir = helper.create_dir(run_dir)
                # Create run stats file: run_stats.csv
                self.run_stats_file = os.path.join(self.run_dir,
                                                   'stats_run.csv')
                helper.write_stats_file(self.run_stats_file,
                                        'episode',
                                        'steps_total', 'steps_mean',
                                        'reward_total', 'reward_mean',
                                        'epsilon')
                self.learner.init_Q(states=self.env.get_all_states(),
                                    how='zero')
                self.learner.set_epsilon(self.params['epsilon'])
                self.learner.set_epsilon(self.params['epsilon'])
                self.run_tests(0, task)
                _logger.info("..... run %s" % str(run))
                self.status = 'training'
                _logger.debug("Current status is %s" % str(self.status))
                for episode in range(1, self.params['episodes'] + 1):
                    self.run_episode(self.env.get_random_state(),
                                     tuple(task['goal_pos']))
                    if episode % self.params['test_interval'] == 0:
                        self.run_tests(episode, task)
                    if self.learner.epsilon > -1 * self.learner.epsilon_change:
                        self.learner.set_epsilon(self.learner.epsilon +
                                                 self.learner.epsilon_change)
                # TODO: Save best Q-table for current run
                helper.plot_run(self.run_dir)
            helper.plot_runs(self.task_dir)
            helper.summarize_runs(self.task_dir)
            helper.plot_task(self.task_dir)
            # TODO: Save best Q-table for current task
            self.status = 'idle'
            _logger.debug("Current status is %s" % str(self.status))
        self.status = 'done'
        _logger.debug("Current status is %s" % str(self.status))

    def run_tests(self, episode, task):
        # TODO: update epsilon
        self.learner.set_epsilon(0.0)
        self.status = 'testing'
        _logger.debug("Current status is %s" % str(self.status))
        self.episode_dir = os.path.join(self.run_dir,
                                        'episode_' + str(episode))
        self.episode_dir = helper.create_dir(self.episode_dir)
        self.test_steps = []
        self.test_rewards = []
        for test_pos in self.params['test_positions']:
            self.init_episode()
            self.run_episode(test_pos,
                             tuple(task['goal_pos']))
            self.test_steps.append(self.steps_in_episode)
            self.test_rewards.append(self.reward_in_episode)
        helper.write_stats_file(self.run_stats_file,
                                episode,
                                sum(self.test_steps),
                                np.mean(self.test_steps),
                                sum(self.test_rewards),
                                np.mean(self.test_rewards),
                                float("{0:.5f}"
                                      .format(self.learner.last_epsilon)))
        self.learner.save_Qs(os.path.join(self.episode_dir,
                                          'Qs.npy'))
        # Make video from random position
        self.status = 'recording'
        _logger.debug("Current status is %s" % str(self.status))
        self.run_episode(self.env.get_random_state(),
                         tuple(task['goal_pos']))
        self.learner.set_epsilon(self.learner.last_epsilon)
        self.status = 'training'
        _logger.debug("Current status is %s" % str(self.status))

if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_Q.yaml')
    exp = Experiment(params_file)
    exp.run()
