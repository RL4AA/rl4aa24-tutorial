import logging.config
import math
import pickle
import random
from enum import Enum

import gymnasium as gym
import numpy as np
from cpymad.madx import Madx
from gymnasium import spaces


# Standard environment for the AWAKE environment, adjusted so it can be used for the
# MAML therefore containing functions for creating and sampling tasks
def generate_optics():
    OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
    MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]

    madx = Madx(stdout=False)
    madx.input("\n".join(MADX_OUT))

    tt43_ini = "../compare/simon_maml/electron_design.mad"

    madx.call(file=tt43_ini, chdir=True)

    madx.use(sequence="tt43", range="#s/plasma_merge")
    quads = {}
    for ele, value in dict(madx.globals).items():
        if "kq" in ele:
            # quads[ele] = value * 0.8
            quads[ele] = value * np.random.uniform(0.5, 1.5, size=None)

    madx.globals.update(quads)

    madx.input(
        "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"  # noqa: E501
    )
    twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

    return twiss_cpymad


class e_trajectory_simENV(gym.Env):
    def __init__(
        self, twiss=[], task={}, train=False, **kwargs
    ):  # each instance of an environment with new generate_optics()
        self.current_action = None
        self.train = train
        self.initial_conditions = []
        self.__version__ = "0.0.1"
        logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_TIME = 50
        self.is_finalized = False
        self.current_episode = -1

        # For internal stats...
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.seed()
        twiss = generate_optics()
        self.twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        self.twiss_correctors = twiss[twiss["keyword"] == "kicker"]

        self.response_scale = 0.5

        self.plane = Plane.horizontal

        self.responseH = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, self.plane
        )
        self.responseV = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, self.plane
        )

        self.positionsH = np.zeros(len(self.twiss_bpms) - 1)  # remove one BPM
        self.settingsH = np.zeros(len(self.twiss_correctors) - 1)  # remove on corrector
        self.positionsV = np.zeros(len(self.twiss_bpms) - 1)
        self.settingsV = np.zeros(len(self.twiss_correctors) - 1)

        high = np.ones(len(self.settingsH))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        high = np.ones(len(self.positionsH))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if "action_scale" in kwargs:
            self.action_scale = kwargs.get("action_scale")
        else:
            self.action_scale = 3e-4
        if "state_scale" in kwargs:
            self.state_scale = kwargs.get("state_scale")
        else:
            self.state_scale = 100
        self.kicks_0 = np.zeros(len(self.settingsH))

        # self.threshold = -0.0001 * self.state_scale  # corresponds to 1.6 mm scaled.
        self.threshold = -0.16  # corresponds to 1.6 mm scaled.

        self._task = task
        self._goal = task
        self._id = task

    def recalculate_response(self):
        self.responseH = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, self.plane
        )
        self.responseV = self._calculate_response(
            self.twiss_bpms, self.twiss_correctors, self.plane
        )

    # MAML specific function, while training samples fresh new tasks and for testing it
    # uses previously saved tasks
    def sample_tasks(self, num_tasks):
        goals = []
        print("Number of tasks: " + str(num_tasks))
        if self.train:
            for i in range(num_tasks):
                twiss = generate_optics()

                twiss_bpms = twiss[twiss["keyword"] == "monitor"]
                twiss_correctors = twiss[twiss["keyword"] == "kicker"]
                responseH = self._calculate_response(
                    twiss_bpms, twiss_correctors, self.plane
                )
                responseV = self._calculate_response(
                    twiss_bpms, twiss_correctors, self.plane
                )
                goals.append([responseH, responseV])
        else:
            with open(
                r"maml_rl/envs/Tasks_data/Tasks", "rb"
            ) as input_file:  # Load in tasks
                goals = pickle.load(input_file)

        # Code used to create sample Tasks
        # with open(
        #     "../AWAKE_MB_META_RL/meta_environment/Tasks_data/Tasks", "wb"
        # ) as fp:  # Pickling
        #     pickle.dump(goals, fp)
        # print("Saved Tasks")

        tasks = [{"goal": goals[i], "id": i} for i in range(num_tasks)]

        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

    def step(self, action):
        # use goal to set variables usually set in init
        self.responseH = self._goal[0]
        self.responseV = self._goal[1]

        state, reward = self._take_action(action)

        self.action_episode_memory[self.current_episode].append(action)

        self.current_steps += 1

        return_reward = reward * self.state_scale

        self.rewards[self.current_episode].append(return_reward)

        return_state = np.array(state * self.state_scale)
        # if return_reward < -1:
        #     return_reward = -2
        #     self.is_finalized = True

        if (return_reward > self.threshold) or self.current_steps > self.MAX_TIME:
            self.is_finalized = True

        return (
            return_state.astype(np.float32),
            return_reward,
            self.is_finalized,
            False,
            {"task": self._id},
        )

    def step_opt(self, action):
        state, reward = self._take_action(action, is_optimisation=True)
        return_reward = reward * self.state_scale
        print(return_reward)
        self.rewards[self.current_episode].append(return_reward)
        return return_reward

    def _take_action(self, action, is_optimisation=False):
        kicks = action * self.action_scale

        state, reward = self._get_state_and_reward(kicks, self.plane, is_optimisation)

        return state, reward

    def _get_reward(self, trajectory):
        rms = np.sqrt(
            np.mean(np.square(trajectory))
        )  # nur first n elements of trajectory
        return rms * (-1.0)

    def _get_state_and_reward(self, kicks, plane, is_optimisation):
        self.TOTAL_COUNTER += 1
        if plane == Plane.horizontal:
            rmatrix = self.responseH

        if plane == Plane.vertical:
            rmatrix = self.responseV

        state = self._calculate_trajectory(rmatrix, self.kicks_0 + kicks)

        if not is_optimisation:
            self.kicks_0 = self.kicks_0 + kicks

        reward = self._get_reward(state)
        return state, reward

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

    def _calculate_trajectory(self, rmatrix, delta_settings):
        delta_settings = np.squeeze(delta_settings)
        return rmatrix.dot(delta_settings)

    def reset(self, **kwargs):
        self.is_finalized = False

        if self.plane == Plane.horizontal:
            self.settingsH = np.random.uniform(-1.0, 1.0, len(self.settingsH))
            self.kicks_0 = self.settingsH * self.action_scale
        if self.plane == Plane.vertical:
            self.settingsV = np.random.uniform(-1.0, 1.0, len(self.settingsV))
            self.kicks_0 = self.settingsV * self.action_scale

        if self.plane == Plane.horizontal:
            # init_positions = np.zeros(len(self.positionsH))  # self.positionsH
            rmatrix = self.responseH

        if self.plane == Plane.vertical:
            # init_positions = np.zeros(len(self.positionsV))  # self.positionsV
            rmatrix = self.responseV

        self.current_episode += 1
        self.current_steps = 0
        self.action_episode_memory.append([])
        self.rewards.append([])
        state = self._calculate_trajectory(rmatrix, self.kicks_0)

        if self.plane == Plane.horizontal:
            self.positionsH = state

        if self.plane == Plane.vertical:
            self.positionsV = state

        # Rescale for agent
        # state = state
        return_initial_state = np.array(state * self.state_scale)
        self.initial_conditions.append([return_initial_state])
        return_value = return_initial_state
        return return_value.astype(np.float32), {"task": self._id}

    def seed(self, seed=None):
        random.seed(seed)

    def setPlane(self, plane):
        if plane == Plane.vertical or plane == Plane.horizontal:
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")


class Plane(Enum):
    horizontal = 0
    vertical = 1
