import logging.config
import math
import os
import pickle
import random
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpymad.madx import Madx
from gymnasium import Wrapper, spaces
from gymnasium.core import WrapperObsType

from maml_rl.envs.helpers import (Awake_Benchmarking_Wrapper, MamlHelpers,
                                  Plane, plot_optimal_policy, plot_results)

# Standard environment for the AWAKE environment,
# adjusted, so it can be used for the MAML therefore containing
# functions for creating and sampling tasks
# def add_key_word_argument_to_env_in_yamml_config(config, **kwargs):
#     with open(config, "r") as f:
#         # Load the configuration from the YAML file
#         config = yaml.safe_load(f)  # using safe_load for better security
#
#     # Update the config with command line arguments
#     # Create a nested 'env-kwargs' dict if it does not exist
#     env_kwargs = config.get("env-kwargs", {})
#     for key in kwargs.keys():
#         # we switch to the no train mode to ensure the predefined task is read
#         env_kwargs[key] = kwargs.get(key)
#
#     config["env-kwargs"] = env_kwargs


class AwakeSteering(gym.Env):
    def __init__(self, twiss=[], task={}, train=False, max_time=1000, **kwargs):
        # print('init env')
        self.init_state_value = None
        self._task = None
        self.kicks_0 = None
        self.__version__ = "0.1"
        logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_TIME = max_time
        self.action_scale = 1e-4
        self.state_scale = 100
        self.threshold = -0.1  # corresponds to 1.6 mm scaled.
        self.reward_scale = 10

        self.train = train
        self.sigma = 0.0

        self.is_finalized = False
        self.current_episode = -1
        self.initial_conditions = []
        self.current_steps = 0

        seed = kwargs.get("seed", None)
        self.seed(seed)

        self.maml_helper = MamlHelpers()
        self.plane = Plane.horizontal
        self.num_random_quads = kwargs.get("num_random_quads", 10)

        self.positionsH = np.zeros(
            len(self.maml_helper.twiss_bpms) - 1
        )  # remove one BPM
        self.settingsH = np.zeros(
            len(self.maml_helper.twiss_correctors) - 1
        )  # remove on corrector
        self.positionsV = np.zeros(len(self.maml_helper.twiss_bpms) - 1)
        self.settingsV = np.zeros(len(self.maml_helper.twiss_correctors) - 1)

        high = np.ones(len(self.settingsH))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        # masking
        self.mask = False
        self.select_items = np.zeros(len(self.settingsH))
        self.select_items[[0, 1, 2, 3, 4, 5]] = 1

        high = np.ones(len(self.positionsH))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if not task:  # If no task is provided, get the origin task
            task = self.maml_helper.get_origin_task()
        self.reset_task(task)

    def step(self, action):
        # action = np.clip(action, -1., 1)

        # Find the maximum absolute value in the action array
        action_abs_max = max(abs(action))

        # Normalize the action array if the maximum absolute value exceeds 1
        if action_abs_max > 1:
            action = action / action_abs_max

        state, reward = self._take_action(action)

        return_state = np.array(state * self.state_scale)
        return_state = np.clip(return_state, -1, 1)
        return_reward = reward * self.state_scale
        self.current_steps += 1

        if (return_reward > self.threshold) or (self.current_steps >= self.MAX_TIME):
            self.is_finalized = True

        violation = np.where(abs(return_state) >= 1)[0]
        if len(violation) > 0:
            return_state[violation[0] :] = np.sign(return_state[violation[0]])
            return_reward = self._get_reward(return_state)
            self.is_finalized = True

        return (
            return_state,
            return_reward * self.reward_scale,
            self.is_finalized,
            False,
            {"task": self._id, "time": self.current_steps},
        )

    def step_opt(self, action):
        state, reward = self._take_action(action, is_optimisation=True)
        return_reward = reward * self.state_scale
        return return_reward

    def _take_action(self, action, is_optimisation=False):
        if self.mask:
            mask = np.clip(
                np.random.randint(2, size=action.shape[-1]) + self.select_items, 0, 1
            )
            action *= mask
        delta_kicks = action * self.action_scale
        state, reward = self._get_state_and_reward(
            delta_kicks, self.plane, is_optimisation
        )
        state += (
            np.random.randn(self.observation_space.shape[0])
            * self.sigma
            / self.state_scale
        )
        return state, reward

    def _get_reward(self, trajectory):
        rms = np.sqrt(np.mean(np.square(trajectory)))
        return -rms

    def _get_state_and_reward(self, kicks, plane, is_optimisation):
        if plane == Plane.horizontal:
            rmatrix = self.responseH
        if plane == Plane.vertical:
            rmatrix = self.responseV

        delta_settings = np.squeeze(self.kicks_0 + kicks)
        state = rmatrix.dot(delta_settings)
        reward = self._get_reward(state)

        if not is_optimisation:
            self.kicks_0 = self.kicks_0 + kicks

        return state, reward

    def find_good_initialisation(self):
        bad_init = True
        counter = 0
        return_initial_state = None
        while bad_init:
            if self.plane == Plane.horizontal:
                kicks_0 = (
                    np.random.uniform(-5, 5, len(self.settingsH)) * self.action_scale
                )
                rmatrix = self.responseH
            if self.plane == Plane.vertical:
                kicks_0 = (
                    np.random.uniform(-5, 5, len(self.settingsV)) * self.action_scale
                )
                rmatrix = self.responseV

            state = rmatrix.dot(kicks_0)
            return_initial_state = np.array(state * self.state_scale)
            return_reward = self._get_reward(return_initial_state)
            if (
                not any(abs(return_initial_state) >= 1)
                and (return_reward < 2 * self.threshold)
                and all(abs(return_initial_state) <= 0.95)
            ):
                bad_init = False

            counter += 1
            if counter > 1000:
                # print('long search')
                break
        # Todo: put at the right place
        self.kicks_0 = kicks_0
        return return_initial_state

    def reset(self, seed: Optional[int] = None, **kwargs):
        self.seed(seed)
        self.is_finalized = False
        self.current_episode += 1
        self.current_steps = 0

        return_initial_state = self.find_good_initialisation()
        self.init_state_value = self._get_reward(return_initial_state)

        return return_initial_state, {}

    def seed(self, seed=None):
        random.seed(seed)

    def setPlane(self, plane: int):
        if plane == Plane.vertical or plane == Plane.horizontal:
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    # MAML specific function, while training samples fresh new tasks
    # and for testing it uses previously saved tasks
    def sample_tasks(self, num_tasks: int):
        tasks = self.maml_helper.sample_tasks(num_tasks, self.num_random_quads)
        return tasks

    def get_origin_task(self, idx: int):
        task = self.maml_helper.get_origin_task(idx=idx)
        return task

    def reset_task(self, task: dict):
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

        self.responseH = self._goal[0]
        self.responseV = self._goal[1]


if __name__ == "__main__":
    # Initialize the environment
    env = Awake_Benchmarking_Wrapper(AwakeSteering())

    # Process each task
    nr_tasks = 5
    for nr in range(nr_tasks):
        env = Awake_Benchmarking_Wrapper(AwakeSteering())
        state = env.reset()[0]
        env.draw_optimal_trajectories(state, 1)
        states_opt, actions_opt, rewards_opt = env.get_optimal_trajectory()

        # Initialize lists for storing results
        states, actions, rewards = [state], [], []
        states_opt_list, actions_opt_list, returns_opt_list = (
            [states_opt],
            [actions_opt],
            [rewards_opt],
        )

        # Execute steps in the environment
        for _ in range(1):  # Using 3 steps as an example
            action = env.action_space.sample()
            next_state, reward, is_finalized, _, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)
            if is_finalized:
                state = env.reset()[0]
                print("reset to: ", state)
                env.draw_optimal_trajectories(state, 1)
            else:
                state = next_state
            states.append(state)

        # Plot results for current task
        plot_results(states, actions, rewards, env, f"Task {nr}")
        plot_optimal_policy(states_opt_list, actions_opt_list, returns_opt_list, env)

    lengths = []
    for _ in range(1000):
        env.reset()
        length = len(env.get_optimal_trajectory()[1]) - 1
        lengths.append(length)

    print(np.mean(lengths))
