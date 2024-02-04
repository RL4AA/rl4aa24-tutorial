import random
from abc import ABC
from enum import Enum

import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from cpymad.madx import Madx
from gymnasium import Env, Wrapper, spaces
from numpy import linalg as LA


def generate_drifting_optics(
    time_step=0, drift_frequency=np.pi * 0.001, drift_amplitude=0.0
):
    OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
    MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]
    madx = Madx(stdout=False)
    madx.input("\n".join(MADX_OUT))
    tt43_ini = "awake_optics/electron_design.mad"
    madx.call(file=tt43_ini, chdir=True)
    madx.use(sequence="tt43", range="#s/plasma_merge")
    quads = {}
    shift = np.sin(drift_frequency * time_step) * drift_amplitude + 1
    i = 0
    for ele, value in dict(madx.globals).items():
        if "kq" in ele:
            i += 1
            # quads[ele] = value * np.random.uniform(0.5, 1.5, size=None)
            if i > 0:
                quads[ele] = value * shift
    madx.globals.update(quads)
    madx.input(
        "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"
    )
    twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

    return twiss_cpymad, shift


class e_trajectory_simENV(Env, ABC):
    """
    Define a simple AWAKE environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, **kwargs):
        # Environment configuration
        self.shift = None
        self.MAX_STEPS = 1000
        self.state_out = False
        self.is_finalized = False
        self.current_steps = 0
        self.total_interactions = 0
        self.traj_return = 0
        self.THRESHOLD_SCALE = 10
        self.plane = Plane.horizontal  # Default plane
        self.sucesses = []

        # Scales and thresholds
        self.action_scale = kwargs.get("scale", 1e-4)
        self.state_scale = 100  # Meters to centimeters
        self.threshold = -0.001 * self.state_scale

        self.drift_amplitude = kwargs.get("drift_amplitude", 0)
        self.drift_frequency = kwargs.get("drift_frequency", 0.001)

        # Initialize system state
        self._initialize_system_state()

        # Action and observation spaces
        self._define_action_observation_spaces()

    def _initialize_system_state(self):
        """
        Initializes the system state, including twiss parameters for BPMs and correctors.
        """
        twiss, _ = generate_drifting_optics(
            drift_amplitude=self.drift_amplitude, drift_frequency=self.drift_frequency
        )
        self._extract_twiss_elements(twiss)
        self.set_system_state(time_step=self.total_interactions)

    def _extract_twiss_elements(self, twiss):
        """
        Extracts BPMs and corrector twiss parameters from the given twiss dataframe.
        """
        self.twiss_bpms = twiss[twiss["keyword"] == "monitor"][1:]
        self.twiss_correctors = twiss[twiss["keyword"] == "kicker"][:-1]
        self.positionsH = np.zeros(len(self.twiss_bpms))
        self.positionsV = np.zeros(len(self.twiss_bpms))
        self.settingsH = np.zeros(len(self.twiss_correctors))
        self.settingsV = np.zeros(len(self.twiss_correctors))
        self.goldenH = np.zeros(len(self.positionsH))
        self.goldenV = np.zeros(len(self.positionsV))
        self.kicks_0 = np.zeros(len(self.settingsH))

    def _define_action_observation_spaces(self):
        """
        Defines the action and observation spaces for the environment.
        """
        action_high = np.ones(len(self.settingsH))
        self.action_space = spaces.Box(
            low=-action_high, high=action_high, dtype=np.float32
        )

        obs_high = np.ones(len(self.positionsH))
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

    def set_system_state(self, time_step):
        """specific function, while training samples fresh new tasks and for testing it uses previously saved tasks"""
        twiss, self.shift = generate_drifting_optics(
            time_step=time_step,
            drift_amplitude=self.drift_amplitude,
            drift_frequency=self.drift_frequency,
        )
        twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        twiss_bpms = twiss_bpms[1:]
        twiss_correctors = twiss[twiss["keyword"] == "kicker"]
        twiss_correctors = twiss_correctors[:-1]
        self.responseH = self._calculate_response(
            twiss_bpms, twiss_correctors, self.plane
        )
        self.responseV = self._calculate_response(
            twiss_bpms, twiss_correctors, self.plane
        )

    def reset(self, **kwargs):
        """
        Reset the state of the environment and return an initial observation.

        Returns:
        -------
        np.ndarray: The initial observation of the space, including the appended value.
        """
        self.is_finalized = False
        self.total_interactions += 1
        self.set_system_state(time_step=self.total_interactions)

        max_attempts = 100  # Prevents infinite loop by setting a max number of initialization attempts
        for _ in range(max_attempts):
            if self.plane == Plane.horizontal:
                settings = self.action_space.sample() * 5
                rmatrix = self.responseH
            elif self.plane == Plane.vertical:
                settings = self.action_space.sample() * 5
                rmatrix = self.responseV
            else:
                raise ValueError(f"Unexpected plane: {self.plane}")

            self.kicks_0 = settings * self.action_scale
            state = self._calculate_trajectory(rmatrix, self.kicks_0)

            if self.plane == Plane.horizontal:
                self.positionsH = state
            else:  # Assuming the only other option is vertical
                self.positionsV = state

            return_initial_state = np.array(state * self.state_scale)
            bad_init = (
                np.any(np.abs(return_initial_state) > 10 * np.abs(self.threshold))
                or LA.norm(return_initial_state) < 0.5
            )

            if not bad_init:
                break
        else:
            # Handle the case where a valid initialization could not be achieved
            raise RuntimeError(
                "Failed to initialize a valid state after multiple attempts."
            )

        self.current_steps = 0
        self.traj_return = 0
        self.sucesses.append(0)
        return return_initial_state, {}

    def step(self, action, reference_position=None):
        """
        Executes one time step within the environment.

        Parameters:
        action: The action to be taken at the current step.
        reference_position (optional): The reference position to compare against, if any.

        Returns:
        A tuple containing the new state, the reward, a boolean indicating whether the episode is finalized, and an empty dictionary.
        """
        self.current_steps += 1
        self.total_interactions += 1
        self.set_system_state(time_step=self.total_interactions)

        action_max = max(abs(action))
        if action_max > 1:
            action /= action_max

        kicks_scaled = action * self.action_scale
        state = self._take_action(kicks_scaled)

        return_state = np.clip(
            np.array(state * self.state_scale),
            -self.THRESHOLD_SCALE * abs(self.threshold),
            self.THRESHOLD_SCALE * abs(self.threshold),
        )

        # Check for "wall hit"
        if (abs(return_state) >= self.THRESHOLD_SCALE * abs(self.threshold)).any():
            self.is_finalized = True
            first_hit_position = np.argmax(
                abs(return_state) >= self.THRESHOLD_SCALE * abs(self.threshold)
            )
            return_state[first_hit_position:] = return_state[first_hit_position]

        return_reward = -np.sqrt(np.mean(np.square(return_state)))

        # Finalize episode on certain conditions
        if self.current_steps >= self.MAX_STEPS or return_reward > self.threshold:
            self.is_finalized = True

        # Accumulate the return
        self.traj_return += return_reward / self.current_steps

        return return_state, return_reward, self.is_finalized, False, {}

    def setGolden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def setPlane(self, plane):
        if plane == Plane.vertical or plane == Plane.horizontal:
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def seed(self, seed):
        np.random.seed(seed)

    def _take_action(self, kicks_scaled):
        self.kicks_0 += kicks_scaled  # Update the kicks directly
        rmatrix, golden = (
            (self.responseH, self.goldenH)
            if self.plane == Plane.horizontal
            else (self.responseV, self.goldenV)
        )
        return self._calculate_trajectory(rmatrix, self.kicks_0)

    def _get_trajectory(self, dkicks, plane):
        if plane == Plane.horizontal:
            rmatrix = self.responseH
            golden = self.goldenH
        elif plane == Plane.vertical:
            rmatrix = self.responseV
            golden = self.goldenV
        trajectory = self._calculate_trajectory(rmatrix, dkicks)
        return trajectory

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        # bpms.pop(0)
        # correctors.pop(-1)

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
                    rmatrix[i][j] = np.sqrt(bpm_beta * corrector_beta) * np.sin(
                        (bpm_mu - corrector_mu) * 2.0 * np.pi
                    )
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def _calculate_trajectory(self, rmatrix, delta_settings):
        delta_settings = np.squeeze(delta_settings)
        return rmatrix.dot(delta_settings)

    def seed(self, seed=None):
        random.seed(seed)


class Plane(Enum):
    horizontal = 0
    vertical = 1


class EpisodeData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_done = False

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def end_episode(self):
        self.is_done = True


class SmartEpisodeTrackerWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self.current_episode = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.current_episode is None:
            self.current_episode = EpisodeData()

        self.current_episode.add_step(observation, action, reward)

        if done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)
            self.current_episode = None

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Start a new episode
        if self.current_episode is not None and not self.current_episode.is_done:
            # End the previous episode if it was not ended properly
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)

        self.current_episode = EpisodeData()
        self.current_episode.add_step(
            observation, None, None
        )  # Initial state with no action or reward
        return observation, info

    def get_episode_data(self, episode_number):
        if episode_number < len(self.episodes):
            return self.episodes[episode_number]
        return None

    def get_all_episodes(self):
        return self.episodes


class EpisodeData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_done = False

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def end_episode(self):
        self.is_done = True


class SmartEpisodeTrackerWithPlottingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self.current_episode = None
        self._setup_plotting()

    def _setup_plotting(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(6, 8), tight_layout=True
        )
        plt.ion()  # Interactive mode for live updates
        self.cumulative_step = 0
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.colors_states = cm.rainbow(np.linspace(0, 1, self.n_states))
        self.colors_actions = cm.rainbow(np.linspace(0, 1, self.n_actions))
        plt.show(block=False)

    def _update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        cumulative_step = 0

        # Function to plot data for an episode
        def plot_episode(episode, start_step):
            if not episode:
                return start_step

            trajectory = (
                np.array(episode.states)
                if episode.states
                else np.zeros((0, self.n_states))
            )
            steps = range(start_step, start_step + len(trajectory))

            for i in range(self.n_states):
                self.ax1.plot(steps, trajectory[:, i], color=self.colors_states[i])

            for i in range(self.n_actions):
                action_values = [
                    action[i] if action is not None and i < len(action) else np.nan
                    for action in episode.actions
                ]
                self.ax2.plot(
                    steps,
                    action_values,
                    color=self.colors_actions[i],
                    ls="--",
                    marker=".",
                )

            return start_step + len(trajectory)

        # Plot data for each completed episode
        for episode in self.episodes:
            cumulative_step = plot_episode(episode, cumulative_step)

        # Plot data for the current (incomplete) episode
        cumulative_step = plot_episode(self.current_episode, cumulative_step)

        self.ax1.set_title("Trajectories for Each Episode")
        self.ax1.set_xlabel("Cumulative Step")
        self.ax1.set_ylabel("State Value")
        self.ax1.grid()

        self.ax2.set_title("Actions for Each Episode")
        self.ax2.set_xlabel("Cumulative Step")
        self.ax2.set_ylabel("Action Value")
        self.ax2.grid()

        # Update legends
        legend_handles_states = [
            mlines.Line2D([], [], color=self.colors_states[i], label=f"State {i + 1}")
            for i in range(self.n_states)
        ]
        legend_handles_actions = [
            mlines.Line2D([], [], color=self.colors_actions[i], label=f"Action {i + 1}")
            for i in range(self.n_actions)
        ]

        # self.ax1.legend(handles=legend_handles_states)
        # self.ax2.legend(handles=legend_handles_actions)
        self.ax1.legend(
            handles=legend_handles_states, loc="upper left", bbox_to_anchor=(1, 1)
        )
        self.ax2.legend(
            handles=legend_handles_actions, loc="upper left", bbox_to_anchor=(1, 1)
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)

        if self.current_episode is None:
            self.current_episode = EpisodeData()

        self.current_episode.add_step(observation, action, reward)

        if done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)
            self.current_episode = None

        self._update_plots()
        return observation, reward, done, False, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        if self.current_episode is not None and not self.current_episode.is_done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)

        self.current_episode = EpisodeData()
        self.current_episode.add_step(
            observation, None, None
        )  # Initial state with no action or reward

        self._update_plots()
        return observation, info


def plot_trajectories_and_actions(env, n_episodes):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    cumulative_step = 0  # To track the cumulative number of steps across episodes

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    colors_states = cm.rainbow(np.linspace(0, 1, n_states))
    colors_actions = cm.rainbow(np.linspace(0, 1, n_actions))

    # Create legend handles
    legend_handles_states = [
        mlines.Line2D([], [], color=colors_states[i], label=f"State {i + 1}")
        for i in range(n_states)
    ]
    legend_handles_actions = [
        mlines.Line2D([], [], color=colors_actions[i], label=f"Action {i + 1}")
        for i in range(n_actions)
    ]

    for _ in range(n_episodes):
        _, _ = env.reset()
        done = False
        trajectory = []
        actions = []

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            trajectory.append(next_state)
            actions.append(action)

        trajectory = np.array(trajectory)
        actions = np.array(actions)

        steps = range(cumulative_step, cumulative_step + len(trajectory))

        for i in range(n_states):
            ax1.plot(steps, trajectory[:, i], color=colors_states[i])
        for i in range(n_actions):
            ax2.plot(steps, actions[:, i], color=colors_actions[i])

        cumulative_step += len(trajectory)

    ax1.set_title("Trajectories for Each Episode")
    ax1.set_xlabel("Cumulative Step")
    ax1.set_ylabel("State Value")
    ax1.legend(handles=legend_handles_states)
    ax1.grid()

    ax2.set_title("Actions for Each Episode")
    ax2.set_xlabel("Cumulative Step")
    ax2.set_ylabel("Action Value")
    ax2.legend(handles=legend_handles_actions)
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    # Initialize the environment
    env = e_trajectory_simENV(drift_amplitude=0.25)
    # plot_trajectories_and_actions(env, 3)  # Plot for 3 episodes

    wrapped_env = SmartEpisodeTrackerWithPlottingWrapper(env)

    for _ in range(10):  # Number of episodes
        obs, _ = wrapped_env.reset()
        done = False
        while not done:
            action = wrapped_env.action_space.sample()
            obs, reward, done, _, _ = wrapped_env.step(action)

    plt.ioff()  # Turn off interactive mode
    plt.show()
