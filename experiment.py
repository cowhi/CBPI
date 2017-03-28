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
        self.current_library = None
        self.library = None

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

    def set_status(self, status, task_name=None, run=None, episode=None):
        self.status = status
        _logger.debug("[T:%s,R:%s,E:%s] %s" %
                      (str(task_name), str(run),
                       str(episode), str(self.status)))

    def init_episode(self):
        self.steps_in_episode = 0
        self.reward_in_episode = 0

    def init_run(self, task_name, run):
        _logger.info("..... Starting run %s" % str(run))
        run_dir = os.path.join(self.task_dir, 'run_' + str(run))
        self.run_dir = helper.create_dir(run_dir)
        # Create run stats file: run_stats.csv
        self.run_stats_file = os.path.join(self.run_dir,
                                           'stats_run.csv')
        helper.write_stats_file(self.run_stats_file,
                                'episode',
                                'steps_total', 'steps_mean',
                                'reward_total', 'reward_mean',
                                'epsilon', 'step_count')

        self.run_steps = 0
        self.init_run_exp(task_name, run)

    def cleanup_run(self, run):
        self.save_best_episode(run)
        helper.delete_dirs(self.run_dir)
        helper.plot_run(self.run_dir)
        self.cleanup_run_exp(self.run_dir)
        _logger.info("..... Finished run %s" % str(run))

    def init_task(self, name):
        _logger.info("##### Starting task %s" % str(name))
        task_dir = os.path.join(self.exp_dir, 'task_' + name)
        self.task_dir = helper.create_dir(task_dir)
        self.init_task_exp(name)

    def cleanup_task(self, name):
        helper.plot_runs(self.task_dir)
        helper.summarize_runs_results(self.task_dir)
        helper.plot_task(self.task_dir)
        self.save_best_run(name)
        self.cleanup_task_exp(name)
        # self.set_s tatus('idle')
        _logger.info("##### Finished task %s" % str(name))

    @abc.abstractmethod
    def init_run_exp(self):
        pass

    @abc.abstractmethod
    def init_task_exp(self, name):
        pass

    @abc.abstractmethod
    def cleanup_task_exp(self, name):
        pass

    def cleanup_run_exp(self):
        pass

    def evaluate_current_library(self):
        pass

    def run_tests(self, task, run, episode):
        self.learner.set_epsilon(0.0)
        self.set_status('testing', task['name'], run, episode)
        self.episode_dir = os.path.join(self.run_dir,
                                        'episode_' + str(episode))
        self.episode_dir = helper.create_dir(self.episode_dir)
        self.test_steps = []
        self.test_rewards = []
        for test_pos in self.params['test_positions']:
            self.init_episode()
            self.run_episode(test_pos,
                             tuple(task['goal_pos']),
                             task['name'])
            self.test_steps.append(self.steps_in_episode)
            self.test_rewards.append(self.reward_in_episode)
        helper.write_stats_file(self.run_stats_file,
                                episode,
                                sum(self.test_steps),
                                np.mean(self.test_steps),
                                sum(self.test_rewards),
                                np.mean(self.test_rewards),
                                float("{0:.5f}"
                                      .format(self.learner.last_epsilon)),
                                self.run_steps)
        self.learner.save_Qs(os.path.join(self.episode_dir,
                                          'Qs.npy'))
        # Make video from random position
        if self.params['visual']:
            self.set_status('recording', task['name'], run, episode)
            self.run_episode(
                self.env.get_random_state(tuple(task['goal_pos'])),
                tuple(task['goal_pos']),
                task['name'])
        self.learner.set_epsilon(self.learner.last_epsilon)

        if not episode % self.params['policy_eval_interval'] == 0 or \
                episode == 0:
            self.set_status('training', task['name'], run, episode)

    def run_episode(self, agent_pos, goal_pos, policy_name=None):
        """
            Function to run a single episode
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
        # TODO: change to give policy instead of library
        action_id = self.learner.get_action(state,
                                            self.current_library,
                                            policy_name,
                                            self.status,
                                            self.params['tau_action'])
        reward = self.env.step(self.env.actions[action_id],
                               self.agent_name)
        state_prime = self.env.get_current_state(self.agent_name)
        if self.status in ['training', 'policy_eval']:
            self.run_steps += 1
        if self.status == 'recording':
            self.env.draw_frame()
            self.env.save_current_frame(self.episode_dir)
        if self.status in ['testing', 'policy_eval']:
            self.steps_in_episode += 1
            self.reward_in_episode += reward
        if self.status == 'policy_eval':
            self.current_library[policy_name]['confidence'] -= \
                self.params['policy_eval_conf_delta']
        if self.status == 'training' and not self.env.episode_ended:
            _logger.debug("%s + %s(%s) = %s + %s" %
                          (str(state),
                           str(action_id),
                           str(self.env.actions[action_id]),
                           str(reward),
                           str(state_prime)))
            _logger.debug("Old Q[%s, %s]: %s" %
                          (str(state[0:2]),
                           str(action_id),
                           str(self.learner.Q[state[0:2], action_id])))
            self.learner.update_Q(state[0:2], action_id,
                                  reward, state_prime[0:2])
            _logger.debug("New Q[%s, %s]: %s" %
                          (str(state[0:2]),
                           str(action_id),
                           str(self.learner.Q[state[0:2], action_id])))
        while not self.env.episode_ended:
            state = state_prime
            action_id = self.learner.get_action(state,
                                                self.current_library,
                                                policy_name,
                                                self.status,
                                                self.params['tau_action'])
            reward = self.env.step(self.env.actions[action_id],
                                   self.agent_name)
            state_prime = self.env.get_current_state(self.agent_name)
            if self.status in ['training', 'policy_eval']:
                self.run_steps += 1
            if self.status == 'recording':
                self.env.draw_frame()
                self.env.save_current_frame(self.episode_dir)
            if self.status in ['testing', 'policy_eval']:
                self.steps_in_episode += 1
                self.reward_in_episode += reward
            if self.status == 'policy_eval':
                if self.current_library[policy_name]['confidence'] > \
                        self.params['policy_eval_conf_stop']:
                    self.current_library[policy_name]['confidence'] -= \
                        self.params['policy_eval_conf_delta']
            if self.status == 'training':
                _logger.debug("%s + %s(%s) = %s + %s" %
                              (str(state),
                               str(action_id),
                               str(self.env.actions[action_id]),
                               str(reward),
                               str(state_prime)))
                _logger.debug("Old Q[%s, %s]: %s" %
                              (str(state[0:2]),
                               str(action_id),
                               str(self.learner.Q[state[0:2], action_id])))
                self.learner.update_Q(state[0:2], action_id,
                                      reward, state_prime[0:2])
                _logger.debug("New Q[%s, %s]: %s" %
                              (str(state[0:2]),
                               str(action_id),
                               str(self.learner.Q[state[0:2], action_id])))
            if self.env.step_count >= self.env.max_steps:
                self.env.episode_ended = True
        if self.status == 'training':
            _logger.debug("End episode")
        if self.env.visual and self.status == 'recording':
            self.env.make_video(self.episode_dir)

    def main(self):
        for task in self.params['tasks']:
            self.init_task(task['name'])
            for run in range(1, self.params['runs'] + 1):
                self.init_run(task['name'], run)
                self.run_tests(task, run, 0)
                for episode in range(1, self.params['episodes'] + 1):
                    self.run_episode(
                        self.env.get_random_state(tuple(task['goal_pos'])),
                        tuple(task['goal_pos']),
                        task['name'])
                    if episode % self.params['test_interval'] == 0:
                        self.run_tests(task, run, episode)
                    if episode % self.params['policy_eval_interval'] == 0:
                        self.evaluate_current_library(task['name'],
                                                      run, episode)
                    # TODO: eval episode ???
                    if self.learner.epsilon > -1 * self.learner.epsilon_change:
                        self.learner.set_epsilon(self.learner.epsilon +
                                                 self.learner.epsilon_change)
                self.cleanup_run(run)
            self.cleanup_task(task['name'])
        _logger.info("Done")

    def save_best_episode(self, run):
        # Save best Q-table for current run
        df = pd.read_csv(os.path.join(self.run_dir, 'stats_run.csv'))
        least_steps_row = df.ix[df['steps_mean'].idxmin()]
        run_best_file = os.path.join(self.run_dir, 'stats_run_best.csv')
        helper.write_stats_file(run_best_file,
                                'run',
                                'episode',
                                'steps_total', 'steps_mean',
                                'reward_total', 'reward_mean',
                                'epsilon')
        helper.write_stats_file(run_best_file,
                                int(run),
                                int(least_steps_row['episode']),
                                least_steps_row['steps_total'],
                                least_steps_row['steps_mean'],
                                least_steps_row['reward_total'],
                                least_steps_row['reward_mean'],
                                least_steps_row['epsilon'])
        helper.copy_file(os.path.join(self.run_dir,
                                      'episode_' +
                                      str(int(least_steps_row['episode'])),
                                      'Qs.npy'),
                         os.path.join(self.run_dir,
                                      'best_Qs.npy'))

    def save_best_run(self, name):
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
        helper.write_stats_file(task_best_file,
                                'task',
                                'run',
                                'episode',
                                'steps_total', 'steps_mean',
                                'reward_total', 'reward_mean',
                                'epsilon')
        helper.write_stats_file(task_best_file,
                                name,
                                int(least_steps_row['run']),
                                int(least_steps_row['episode']),
                                least_steps_row['steps_total'],
                                least_steps_row['steps_mean'],
                                least_steps_row['reward_total'],
                                least_steps_row['reward_mean'],
                                least_steps_row['epsilon'])
        helper.copy_file(os.path.join(self.task_dir,
                                      'run_' +
                                      str(int(least_steps_row['run'])),
                                      'best_Qs.npy'),
                         os.path.join(self.task_dir,
                                      'best_Qs.npy'))


if __name__ == "__main__":
    params_file = os.path.join(os.getcwd(),
                               'params_Q.yaml')
    exp = Experiment(params_file)
    exp.main()
