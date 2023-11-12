import os
import pickle
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import gym
from gym import spaces
from matplotlib import pyplot as plt


class RecordStatisticsWrapper(gym.Wrapper):
    def __init__(self, env, experiment='default', deque_size=1000):
        super().__init__(env)
        self.current_history = None
        self.experiment = experiment
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.histories_queue = deque(maxlen=deque_size)
        self.no_limit = False

        self.save_frequency = 50
        self.total_counter = 0
        self.storage_queue = deque(maxlen=self.save_frequency)

        self.geneterate_plot_vars()
        self.generate_plots()


        # high = 1 * np.ones(len(self.bpmsH.elements)+1)
        # low = (-1) * high
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # print('wrapper')

    def reset(self, **kwargs):
        self.total_counter += 1
        if 'no_limit' in kwargs:
            self.no_limit = kwargs.get('no_limit')
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        # self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_lengths = 0

        self.current_history = []
        self.current_history.append([np.array(observations), np.nan * np.empty(self.action_space.shape), np.nan])
        self.history_management([self.episode_count, self.episode_lengths,
                                 [observations, np.nan * np.empty(self.action_space.shape), np.nan]], info='reset')
        self.update_plot_vars(init=True, update_vars=observations)

        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.total_counter += 1

        self.current_history.append([np.array(observations), action, rewards])

        self.history_management([self.episode_count, self.episode_lengths, [observations, action, rewards]])

        self.update_plot_vars()
        self.update_plots(action=action, return_state=observations, reward=rewards)

        infos = [infos]
        dones = [dones]

        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                # episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    # "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                # self.length_queue.append(episode_length)
                self.histories_queue.append(self.current_history)
                self.history_management([self.episode_count, self.episode_lengths,
                                         [observations, action, rewards]], info='done')
                if not self.no_limit:
                    self.episode_count += 1
                    self.episode_returns[i] = 0
                    # self.episode_lengths[i] = 0

        return (
            observations,
            rewards,
            dones[0],
            infos[0],
        )

    def history_management(self, data, info=False):
        if info == 'reset':
            self.storage_queue.clear()
            self.storage_queue.append(data)
        elif info == 'done':
            self.storage_queue.append(data)
            self.save_state_to_disk(self.storage_queue)
        else:
            self.storage_queue.append(data)

        if self.total_counter % self.save_frequency == 0:
            self.save_state_to_disk(self.storage_queue)

    def save_state_to_disk(self, my_queue):
        if not os.path.exists(self.experiment):
            os.makedirs(self.experiment)

        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")[:-3]

        with open(os.path.join(self.experiment, date_time + ".obj"), "wb+") as queue_save_file:
            pickle.dump(my_queue, queue_save_file)
        print(f'saving...{self.total_counter}')

    def update_plots(self, return_state, action, reward):
        # print(reward, 'wrapper ____' * 20)
        # self.rews[self.total_episodes_wrapper] += reward/self.lens[-1]
        self.rews[self.total_episodes_wrapper] = reward

        self.current_traj_clipped.set_ydata(return_state[:-1])
        self.previous_traj.set_ydata(self.previous_state[:-1])

        self.ax1.set_title(
            f'ep:{self.total_episodes_wrapper} ep steps {self.current_interactions_wrapper} total: {self.total_interactions_wrapper},'
            f'rew: {reward}')

        self.rew_traj.set_data(np.array(range(self.total_episodes_wrapper + 1)), np.array(self.rews))
        self.len_traj.set_data(np.array(range(self.total_episodes_wrapper + 1)), np.array(self.lens))
        self.done_traj.set_data(np.array(range(self.total_episodes_wrapper + 1)), np.array(self.sucesses))

        self.ax2.set_xlim(0, self.total_episodes_wrapper)
        self.ax2_twin.set_ylim(0, np.max(self.lens))
        self.actions.append(action)
        self.ax3.clear()
        self.ax3.plot(self.actions)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.previous_state = return_state

    def generate_plots(self):
        # plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        self.x_line = range(self.observation_space.shape[0]-1)
        # self.current_traj, = self.ax.plot(self.x_line, np.zeros(self.action_space.shape[0]), 'b-')
        self.current_traj_clipped, = self.ax1.plot(self.x_line, np.zeros(self.observation_space.shape[0]-1), 'r-')
        self.previous_traj, = self.ax1.plot(self.x_line, np.zeros(self.observation_space.shape[0]-1), 'b:')
        self.ax1.axhline(10 * abs(self.threshold))
        self.ax1.axhline(-10 * abs(self.threshold))
        self.ax1.set_ylim(-2, 2)

        self.rew_traj, = self.ax2.plot(0, 0, 'g-')
        self.ax2_twin = self.ax2.twinx()
        self.len_traj, = self.ax2_twin.plot(0, 0, 'b-')
        self.done_traj, = self.ax2_twin.step(0, 0, 'lime')
        self.ax2.set_ylim(-1, .1)
        # self.ax2_twin.set_ylim(0, self.MAX_Steps + 1)
        self.ax2_twin.set_ylim(0, 10)
        self.fig.suptitle('Wrapper')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def geneterate_plot_vars(self):
        self.rews = []
        self.actions = []
        self.lens = []
        self.sucesses = []
        self.current_interactions_wrapper = 0
        self.total_interactions_wrapper = 0
        self.total_episodes_wrapper = -1

    def update_plot_vars(self, init=False, update_vars=None):
        if init:
            self.total_interactions_wrapper += 1
            self.current_interactions_wrapper = 0
            self.total_episodes_wrapper += 1
            self.rews.append(0)
            self.lens.append(0)
            self.sucesses.append(0)
            self.previous_state = update_vars
        else:
            self.total_interactions_wrapper += 1
            self.current_interactions_wrapper += 1
            self.lens[self.total_episodes_wrapper] = self.current_interactions_wrapper
