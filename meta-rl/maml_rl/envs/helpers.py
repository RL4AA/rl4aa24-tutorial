import math
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpymad.madx import Madx
from gymnasium import Wrapper
from gymnasium.core import WrapperObsType


class Plane(Enum):
    horizontal = 0
    vertical = 1


class MamlHelpers:
    # init
    def __init__(self):
        self.twiss = self._generate_optics()
        self.response_scale = 0.5

        self.twiss_bpms = self.twiss[self.twiss["keyword"] == "monitor"]
        self.twiss_correctors = self.twiss[self.twiss["keyword"] == "kicker"]

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        bpms.pop(0)
        correctors.pop(-1)
        rmatrix = np.zeros((len(bpms), len(correctors)))
        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if plane == Plane.horizontal:
                    bpm_beta = bpmsTwiss.betx[bpm]
                    corrector_beta = correctorsTwiss.betx[corrector]
                    bpm_mu = bpmsTwiss.mux[bpm]
                    corrector_mu = correctorsTwiss.mux[corrector]
                else:
                    bpm_beta = bpmsTwiss.bety[bpm]
                    corrector_beta = correctorsTwiss.bety[corrector]
                    bpm_mu = bpmsTwiss.muy[bpm]
                    corrector_mu = correctorsTwiss.muy[corrector]

                if bpm_mu > corrector_mu:
                    rmatrix[i][j] = (
                        math.sqrt(bpm_beta * corrector_beta)
                        * math.sin((bpm_mu - corrector_mu) * 2.0 * math.pi)
                        * self.response_scale
                    )
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def generate_optics(self, randomize: bool = True, num_random_quads: int = 10):
        twiss = self._generate_optics(
            randomize=randomize, num_random_quads=num_random_quads
        )
        twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        twiss_correctors = twiss[twiss["keyword"] == "kicker"]
        responseH = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.horizontal
        )
        responseV = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.vertical
        )
        return responseH, responseV

    def recalculate_response(self):
        responseH = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, Plane.horizontal
        )
        responseV = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, Plane.vertical
        )
        return responseH, responseV

    def _generate_optics(self, randomize: bool = False, num_random_quads: int = 10):
        OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
        MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]
        madx = Madx(stdout=False)
        madx.input("\n".join(MADX_OUT))
        tt43_ini = "maml_rl/envs/electron_design.mad"
        madx.call(file=tt43_ini, chdir=True)
        madx.use(sequence="tt43", range="#s/plasma_merge")
        quads = {}
        variation_range = (0.75, 1.25)

        if randomize:  # randomize quads
            # get all quads settings
            for ele, value in dict(madx.globals).items():
                if "kq" in ele:
                    # quads[ele] = value * 0.8
                    quads[ele] = value
            n_quads = len(quads)
            random_factors = np.ones(n_quads)
            num_random_quads = min(num_random_quads, n_quads)
            # only randomize the last num_random_quads quadrupoles
            random_factors[-num_random_quads:] = np.random.uniform(
                variation_range[0], variation_range[1], num_random_quads
            )
            for i, (ele, value) in enumerate(quads.items()):
                quads[ele] = value * random_factors[i]

        madx.globals.update(quads)
        madx.input(
            "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,"
            + "BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"
        )
        twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

        return twiss_cpymad

    def sample_tasks(self, num_tasks: int, num_random_quads: int = 10):
        # Generate goals using list comprehension for more concise code
        goals = [
            self.generate_optics(randomize=True, num_random_quads=num_random_quads)
            for _ in range(num_tasks)
        ]

        # Create tasks with goals and corresponding IDs using list comprehension
        tasks = [{"goal": goal, "id": idx} for idx, goal in enumerate(goals)]
        return tasks

    def get_origin_task(self, idx=0):
        # Generate goals using list comprehension for more concise code
        goal = self.generate_optics(randomize=False)
        # Create tasks with goals and corresponding IDs using list comprehension
        task = {"goal": goal, "id": idx}
        return task


class Awake_Benchmarking_Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.invV = None
        self.invH = None
        self.optimal_rewards = None
        self.optimal_actions = None
        self.optimal_states = None

    def reset(self) -> tuple[WrapperObsType, dict[str, Any]]:
        #     print('reset', self.current_steps, self._get_reward(return_value))
        return_initial_state, _ = self.env.reset()

        self.invH, self.invV = (
            np.linalg.inv(self.env.responseH / 100) / 100,
            np.linalg.inv(self.env.responseV / 100) / 100,
        )
        (
            self.optimal_states,
            self.optimal_actions,
            self.optimal_rewards,
        ) = self._get_optimal_trajectory(return_initial_state)
        return return_initial_state, {}

    def policy_optimal(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        action = -invrmatrix.dot(state * self.env.state_scale)
        # action = np.clip(action, -1, 1)
        action_abs_max = max(abs(action))
        if action_abs_max > 1:
            action /= action_abs_max
        return action

    def get_k_for_state(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        k = invrmatrix.dot(state * self.env.state_scale) * self.env.action_scale
        return k

    def get_optimal_trajectory(self):
        return self.optimal_states, self.optimal_actions, self.optimal_rewards

    def _get_optimal_trajectory(self, init_state):
        max_iterations = 25
        states = [init_state]
        actions = []
        # Todo: reward scaling
        rewards = [self.env._get_reward(init_state) * self.env.reward_scale]

        self.env.kicks_0_opt = self.env.kicks_0.copy()
        self.env.kicks_0 = self.get_k_for_state(init_state)
        self.env.is_finalized = False

        for i in range(max_iterations):
            action = self.policy_optimal(states[i])
            actions.append(action)
            state, reward, is_finalized, _, _ = self.env.step(action)

            states.append(state)
            rewards.append(reward)

            if is_finalized:
                break

        if i < max_iterations - 1:
            # nan_state = [np.nan] * self.env.observation_space.shape[-1]
            # nan_action = [np.nan] * self.env.action_space.shape[-1]
            # states[i + 2:] = [nan_state] * (max_iterations - i - 1)
            # actions[i + 1:] = [nan_action] * (max_iterations - i - 1)
            states.append([np.nan] * self.env.observation_space.shape[-1])
            actions.append([np.nan] * self.env.action_space.shape[-1])
            rewards.append(np.nan)

        self.env.kicks_0 = self.env.kicks_0_opt.copy()
        self.env.is_finalized = False
        return states, actions, rewards

    def draw_optimal_trajectories(self, init_state, nr_trajectories=5):
        states_frames, actions_frames, rewards_frames = [], [], []
        len_mean = []

        for _ in range(nr_trajectories):
            states, actions, rewards = self._get_optimal_trajectory(init_state)
            states_frames.append(pd.DataFrame(states))
            actions_frames.append(pd.DataFrame(actions))
            rewards_frames.append(pd.DataFrame(rewards))
            # actions end with np.nan to find episode ends
            len_mean.append(len(actions) - 1)

        mean_length = np.mean(len_mean)
        # print(mean_length)

        states_df = pd.concat(states_frames, ignore_index=True)
        actions_df = pd.concat(actions_frames, ignore_index=True)
        rewards_df = pd.concat(rewards_frames, ignore_index=True)

        fig, axs = plt.subplots(3, figsize=(10, 10))
        for df, ax in zip([states_df, rewards_df, actions_df], axs):
            df.plot(ax=ax)

        plt.suptitle(f"Mean Length of Episodes: {mean_length}")
        plt.tight_layout()
        plt.show()
        plt.pause(1)


# Helper functions for plotting
def plot_results(states, actions, rewards, env, title):
    fig, axs = plt.subplots(3)
    axs[0].plot(states)
    axs[0].set_title("States")
    axs[1].plot(actions)
    axs[1].set_title("Actions")
    axs[2].plot(rewards)
    axs[2].set_title("Rewards")
    axs[2].axhline(env.unwrapped.threshold, c="r")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_optimal_policy(states_opt_list, actions_opt_list, returns_opt_list, env):
    states_df = pd.concat(
        [pd.DataFrame(states) for states in states_opt_list], ignore_index=True
    )
    actions_df = pd.concat(
        [pd.DataFrame(actions) for actions in actions_opt_list], ignore_index=True
    )
    returns_df = pd.concat(
        [pd.DataFrame(rewards) for rewards in returns_opt_list], ignore_index=True
    )
    episode_lengths = [len(rewards) - 1 for rewards in returns_opt_list]

    fig, axs = plt.subplots(4, figsize=(10, 10))
    states_df.plot(ax=axs[0])
    axs[0].set_title("States")
    actions_df.plot(ax=axs[1])
    axs[1].set_title("Actions")
    returns_df.plot(ax=axs[2])
    axs[2].axhline(env.unwrapped.threshold, c="r")
    axs[2].set_title("Return")
    axs[3].plot(episode_lengths)
    axs[3].set_title("Length")
    plt.suptitle("Optimal Policy")
    plt.tight_layout()
    plt.show()
