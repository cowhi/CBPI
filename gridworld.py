#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Needed for visualization
# import Tkinter as tk
# from Python Image Library (PIL) for image object
from PIL import Image
from PIL import ImageDraw
# others
import os
import cv2
import glob
import numpy as np

# some color constants for PIL
white = (255, 255, 255)  # room
yellow = (255, 255, 0)  # corridor
black = (0, 0, 0)  # walls
blue = (0, 0, 255)  # doors
red = (255, 0, 0)  # death
green = (0, 128, 0)  #
blue_light = (0, 191, 255)  # door
green_light = (0, 255, 0)  # reward
yellow_light = (255, 255, 224)  # corridor


class Gridworld(object):
    def __init__(self, grid, cell_width=10, cell_height=10,
                 uncertainty=0.0, step_size=1, step_penalty=0,
                 max_steps=100, visual=False, rng=np.random.RandomState(1)):
        self.rng = rng
        # create the image object for learning only if necessary
        self.visual = visual
        with open(grid) as f:
            self.lines = [line.rstrip() for line in f]
        # self.lines = [line.rstrip() for line in open(grid)]
        self.row_count = len(self.lines)
        self.column_count = len(self.lines[0])
        # Initialize objects and possible states
        self.agents = {}
        self.goals = {}
        self.all_states = self.lookup_states()
        # set possible actions
        self.actions = ['west',  # 0: left
                        'east',  # 1: right
                        'north',  # 2: up
                        'south']  # 3: down
        # environment settings
        self.step_size = step_size
        self.step_penalty = step_penalty
        self.step_uncertainty = uncertainty
        self.max_steps = max_steps
        # self.reward_goal = 1
        self.goal_reached = False
        self.episode_ended = False
        self.reached_goal_name = None
        if self.visual:
            # set all map information
            self.cell_width = cell_width
            self.cell_height = cell_height
            self.map_width = self.column_count * self.cell_width
            self.map_height = self.row_count * self.cell_height
            self.visual = visual
            self.image_frame = Image.new("RGB",
                                         (self.map_width, self.map_height),
                                         white)
            self.image = ImageDraw.Draw(self.image_frame)

    """
        Funktions to draw the canvas and prepare the image.
    """

    def draw_frame(self):
        self.draw_map()
        self.draw_goals()
        self.draw_agents()

    def draw_map(self):
        row = 0
        for line in self.lines:
            column = 0
            for symbol in line:
                x1 = column * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                column += 1
                if symbol == "-":
                    self.image.rectangle([x1, y1, x2, y2], black)
                if symbol == "r":
                    # self.image.rectangle([x1, y1, x2, y2], white)
                    self.image.rectangle([x1, y1, x2, y2], yellow_light)
                if symbol == "c":
                    # self.image.rectangle([x1, y1, x2, y2], yellow)
                    self.image.rectangle([x1, y1, x2, y2], white)
                if symbol == "d":
                    # self.image.rectangle([x1, y1, x2, y2], blue_light)
                    self.image.rectangle([x1, y1, x2, y2], white)
            row += 1

    def draw_agents(self):
        for agent_coords in self.agents.values():
            x1 = int(agent_coords[0]) * self.cell_width
            y1 = int(agent_coords[1]) * self.cell_height
            x2 = x1 + self.cell_width
            y2 = y1 + self.cell_height
            self.image.ellipse([x1, y1, x2, y2], blue_light)
            # self.image.ellipse([x1, y1, x2, y2], white)

    def draw_goals(self):
        for goal_coords, goal_values in self.goals.items():
            x1 = goal_coords[0] * self.cell_width
            y1 = goal_coords[1] * self.cell_height
            x2 = x1 + self.cell_width
            y2 = y1 + self.cell_height
            if goal_values[0] < 0:
                self.image.rectangle([x1, y1, x2, y2], red)
            else:
                self.image.rectangle([x1, y1, x2, y2], green_light)

    """
        Funktions to control the environment.
    """

    def lookup_states(self):
        possible_states = []
        row = 0
        for line in self.lines:
            column = 0
            for symbol in line:
                if not symbol == '-':
                    possible_states.append((column, row))
                column += 1
            row += 1
        return possible_states

    def add_goal(self, pos, name='goal', reward=1.0):
        self.goals[pos] = (reward, name)

    def add_agent(self, pos, name='agent'):
        self.agents[name] = [pos[0], pos[1]]

    def get_random_state(self, goal_state=None):
        states = self.all_states[:]
        try:
            states.remove(goal_state)
        except:
            pass
        return states[self.rng.randint(0, len(states))]

    def run_episode(self, directory='test'):
        """
            Function to perform a test run
        """
        self.reset_env()
        self.draw_frame()
        self.save_current_frame(directory)
        action_id = self.rng.random_integers(0, self.action_count - 1)
        reward = self.step(self.actions[action_id])
        self.draw_frame()
        self.save_current_frame(directory)
        '''
        print('Step: %d, Action: %s, Reward: %s' % (self.step_count,
                                                    self.actions[action_id],
                                                    str(reward)))
        '''
        while not self.episode_ended:
            action_id = self.rng.random_integers(0, self.action_count - 1)
            reward = self.step(self.actions[action_id])
            self.draw_frame()
            self.save_current_frame(directory)
            '''
            print('Step: %d, Action: %s, '
                  'Reward: %s' % (self.step_count,
                                  self.actions[action_id],
                                  str(reward)))
            '''
            if reward == 0.0:
                pass
            if self.step_count >= self.max_steps:
                self.episode_ended = True
        if self.visual:
            self.make_video(directory)

    def reset_env(self):
        self.step_count = 0
        self.episode_ended = False
        self.goal_reached = False
        self.goals = {}
        self.agents = {}
        self.reached_goal_name = None
        self.reward_current = 0
        self.reward_total = 0

    def step(self, action, agent='agent'):
        """
            Simply updates the coordinates of the agent for the
            respective action if the proposed direction does not lead
            into a wall.
        """
        self.step_count += 1
        step = self.step_size + self.get_slip()
        can_move = True
        if action == 'west':
            for x in range(int(self.agents[agent][0] - step),
                           int(self.agents[agent][0]) + 1):
                try:
                    if self.lines[int(self.agents[agent][1])][x] == '-':
                        can_move = False
                except IndexError:
                    can_move = False
            if can_move:
                self.agents[agent][0] -= step
        if action == 'east':
            for x in range(int(self.agents[agent][0]),
                           int(self.agents[agent][0] + step) + 1):
                try:
                    if self.lines[int(self.agents[agent][1])][x] == '-':
                        can_move = False
                except IndexError:
                    can_move = False
            if can_move:
                self.agents[agent][0] += step
        if action == 'north':
            for y in range(int(self.agents[agent][1] - step),
                           int(self.agents[agent][1]) + 1):
                try:
                    if self.lines[y][int(self.agents[agent][0])] == '-':
                        can_move = False
                except IndexError:
                    can_move = False
            if can_move:
                self.agents[agent][1] -= step
        if action == 'south':
            for y in range(int(self.agents[agent][1]),
                           int(self.agents[agent][1] + step) + 1):
                try:
                    if self.lines[y][int(self.agents[agent][0])] == '-':
                        can_move = False
                except IndexError:
                    can_move = False
            if can_move:
                self.agents[agent][1] += step
        self.reward_current = self.get_reward_current()
        self.reward_total += self.reward_current
        return self.reward_current

    def get_slip(self):
        return self.rng.uniform(-1 * self.step_uncertainty,
                                self.step_uncertainty)

    def get_reward_current(self):
        for agent_name, agent_values in self.agents.items():
            for goal_coords, goal_values in self.goals.items():
                if int(agent_values[0]) == goal_coords[0] \
                        and int(agent_values[1]) == goal_coords[1]:
                    self.episode_ended = True
                    self.goal_reached = True
                    self.reached_goal_name = goal_values[1]
                    return goal_values[0]
        return self.step_penalty

    def get_reward_total(self):
        return self.reward_total

    def get_goal_status(self):
        return self.goal_reached

    def save_current_frame(self, directory):
        name = 'step_{0:0>6}'.format(self.step_count) + '.png'
        self.image_frame.save(os.path.join(directory, name))

    def get_current_state(self, agent):
        # if self.visual:
        #    return (int(self.agents[agent][0]), int(self.agents[agent][1]),
        #            np.asarray(self.image_frame))
        # else:
        return (int(self.agents[agent][0]), int(self.agents[agent][1]))

    def get_all_states(self):
        return self.all_states

    def get_map_size(self):
        return (self.row_count, self.column_count)

    """
        Some helper functions to better use the environment.
    """

    def make_video(self, path_to_dir):
        # find all images files
        frames = glob.glob(os.path.join(path_to_dir) + '/*.png')
        start_frame = cv2.imread(frames[0])
        # get image data
        video_height, video_width, layers = start_frame.shape
        # open video object
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(os.path.join(path_to_dir, 'video.avi'),
                                fourcc,
                                6,
                                (video_width, video_height))
        # write every image to video object
        for frame in frames:
            video.write(cv2.imread(frame))
        # wrap video up
        cv2.destroyAllWindows()
        video.release()
        # remove all images exept first and last
        for i in range(1, len(frames)-1):
            os.remove(frames[i])


if __name__ == "__main__":
    env = Gridworld('./maps/PolicyReuse2006', visual=True)
    env.add_agent((3, 3))
    env.add_goal((1, 1))
    env.run_episode()


# TODO:
# get state as image or agent coords
# add multiagent support
