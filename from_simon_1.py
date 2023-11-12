import logging.config
import random
from abc import ABC
from enum import Enum

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpymad.madx import Madx
# 3rd party modules
# from bayes_opt import BayesianOptimization
from gym import spaces

import scipy.optimize as opt

from Application import twissReader
# from mbrllib.mbrl.env.Application import twissReader
# from Application import twissReader
from record_statistics_wrapper_rew import RecordStatisticsWrapper

from numpy import linalg as LA


def generate_drifting_optics(time_step=0, drift_frequency=np.pi * 0.001, drift_amplitude=0.0):
    OPTIONS = ['WARN']  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
    MADX_OUT = [f'option, -{ele};' for ele in OPTIONS]
    madx = Madx(stdout=False)
    madx.input('\n'.join(MADX_OUT))
    tt43_ini = "Application/electron_design.mad"
    madx.call(file=tt43_ini, chdir=True)
    madx.use(sequence="tt43", range='#s/plasma_merge')
    quads = {}
    shift = np.sin(drift_frequency * time_step) * drift_amplitude + 1
    print('shift', shift)
    i = 0
    for ele, value in dict(madx.globals).items():
        if 'kq' in ele:
            i += 1
            # quads[ele] = value * np.random.uniform(0.5, 1.5, size=None)
            if i > 20:
                quads[ele] = value * shift
                print(ele)
    madx.globals.update(quads)
    madx.input('initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;')
    twiss_cpymad = madx.twiss(beta0='initbeta0').dframe()

    return twiss_cpymad


class e_trajectory_simENV(gym.Env, ABC):
    """
    Define a simple AWAKE environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, **kwargs):

        self.state_out = False
        # self.__version__ = "0.0.1"
        # logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_Steps = 1000
        self.is_finalized = False
        self.total_interactions = 0
        self.current_steps = 0
        self.total_episodes = -1
        self.traj_return = 0

        self.plane = Plane.horizontal  # hard coded

        self.seed()
        twiss = generate_drifting_optics()
        self.twiss_bpms = twiss[twiss['keyword'] == 'monitor']
        self.twiss_bpms = self.twiss_bpms[1:]
        self.twiss_correctors = twiss[twiss['keyword'] == 'kicker']
        self.twiss_correctors = self.twiss_correctors[:-1]

        self.set_system_state(time_step=self.total_interactions)

        # self.twissH, self.twissV = twissReader.readAWAKEelectronTwiss()

        # self.bpmsH = self.twissH.getElements("BP")
        # self.bpmsV = self.twissV.getElements("BP")

        # self.bpmsV.remove(0)
        # self.bpmsH.remove(0)

        # self.correctorsH = self.twissH.getElements("MCA")
        # self.correctorsV = self.twissV.getElements("MCA")

        # self.correctorsV.remove(-1)
        # self.correctorsH.remove(-1)

        # print((self.bpmsH.getNames()))
        # print(len(self.correctorsH.getNames()))

        # for _ in range(1, 9):
        #     self.bpmsV.remove(0)
        #     self.bpmsH.remove(0)
        #     self.correctorsV.remove(0)
        #     self.correctorsH.remove(0)

        # print((self.bpmsH.getNames()))
        # print(len(self.correctorsH.getNames()))

        # self.responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        # self.responseV = self._calculate_response(self.bpmsV, self.correctorsV)
        self.positionsH = np.zeros(len(self.twiss_bpms))
        self.settingsH = np.zeros(len(self.twiss_correctors))
        self.positionsV = np.zeros(len(self.twiss_bpms))
        self.settingsV = np.zeros(len(self.twiss_correctors))

        # self.positionsH = np.zeros(len(self.bpmsH.elements))
        # self.settingsH = np.zeros(len(self.correctorsH.elements))
        # self.positionsV = np.zeros(len(self.bpmsV.elements))
        # self.settingsV = np.zeros(len(self.correctorsV.elements))
        # self.goldenH = np.zeros(len(self.bpmsV.elements))
        # self.goldenV = np.zeros(len(self.bpmsV.elements))

        self.goldenH = np.zeros(len(self.positionsH))
        self.goldenV = np.zeros(len(self.positionsV))

        # high = 1 * np.ones(len(self.correctorsH.elements))
        high = 1 * np.ones(len(self.settingsH))
        low = (-1) * high

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        high = 1 * np.ones(len(self.positionsH))
        low = (-1) * high
        self.observation_space = spaces.Box(low=np.append(low, 0), high=np.append(high, 1), dtype=np.float32)

        if 'scale' in kwargs:
            self.action_scale = kwargs.get('scale')
        else:
            self.action_scale = 1e-4

        # init position of the magnets set to 0 but changed in the reset function
        self.kicks_0 = np.zeros(len(self.settingsH))

        self.state_scale = 100  # Meters to cms as given from BPMs in the measurement later on

        # the threshold for the ending of the episodes
        self.threshold = -0.001 * self.state_scale

        self.sucesses = []

    def set_system_state(self, time_step):
        '''specific function, while training samples fresh new tasks and for testing it uses previously saved tasks'''
        twiss = generate_drifting_optics(time_step=time_step)
        twiss_bpms = twiss[twiss['keyword'] == 'monitor']
        twiss_bpms = twiss_bpms[1:]
        twiss_correctors = twiss[twiss['keyword'] == 'kicker']
        twiss_correctors = twiss_correctors[:-1]
        self.responseH = self._calculate_response(twiss_bpms, twiss_correctors, self.plane)
        self.responseV = self._calculate_response(twiss_bpms, twiss_correctors, self.plane)

    def reset(self, **kwargs):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.is_finalized = False
        # self.current_steps = 0
        self.total_interactions += 1
        self.set_system_state(time_step=self.total_interactions)

        bad_init = True
        while bad_init:
            if (self.plane == Plane.horizontal):
                # Uniformly randomize actions
                self.settingsH = self.action_space.sample() * 5
                # scale actions to problem
                self.kicks_0 = self.settingsH * self.action_scale
                rmatrix = self.responseH

            if (self.plane == Plane.vertical):
                # Uniformly randomize actions
                self.settingsV = self.action_space.sample() * 5
                # scale actions to problem
                self.kicks_0 = self.settingsV * self.action_scale
                rmatrix = self.responseV

            state = self._calculate_trajectory(rmatrix, self.kicks_0)
            if (self.plane == Plane.horizontal):
                self.positionsH = state
            if (self.plane == Plane.horizontal):
                self.positionsV = state

            return_initial_state = np.array(state * self.state_scale)
            bad_init = any(abs(return_initial_state) > 10 * abs(self.threshold)) or LA.norm(return_initial_state) < 0.5

        #
        self.current_steps = 0
        self.traj_return = 0
        # self.total_episodes += 1
        # self.rews.append(0)
        # self.lens.append(0)
        self.sucesses.append(0)

        # print('reset:', self.total_episodes, self.total_interactions, self.current_interactions)
        # print('init stat: ', return_initial_state)

        # self.previous_state = return_initial_state
        print('************* reset  ' * 5)
        # print(np.append(return_initial_state, 0))
        return np.append(return_initial_state, 0.0)

    def step(self, action, reference_position=None):
        print('step...' * 20)
        self.current_steps += 1

        self.total_interactions += 1
        self.set_system_state(time_step=self.total_interactions)

        # if self.current_steps % 5 == 0:
        # action = self.action_space.sample()*3
        # print('** rand '*5)
        kicks_scaled = action * self.action_scale
        state = self._take_action(kicks_scaled)

        golden_trajectory = 0
        # scaling and clipping
        # self.previous_traj.set_ydata(self.previous_state)
        # self.current_traj.set_ydata(np.array(state * self.state_scale))

        return_state = np.clip(np.array(state * self.state_scale), -10 * abs(self.threshold), 10 * abs(self.threshold))

        # we hit the wall
        if (abs(return_state) >= abs(10 * self.threshold)).any():
            self.is_finalized = True
            first_hit_position = np.argmax(abs(return_state) >= abs(10 * self.threshold))
            # return_state[first_hit_position:] = return_state[first_hit_position]
            for i in range(first_hit_position, len(return_state)):
                return_state[i] = return_state[first_hit_position]  # *(i-first_hit_position+1)
            # if it would continue till the end of the episode
            return_reward = -np.sqrt(np.mean(np.square(return_state - golden_trajectory * self.state_scale)))  # *\
            # (self.MAX_Steps+1-self.current_steps)
        else:
            return_reward = -np.sqrt(np.mean(np.square(return_state - golden_trajectory * self.state_scale)))
        self.state_out = False
        # Check if episode length is max
        if self.current_steps >= self.MAX_Steps:
            self.is_finalized = True
        # Check if target is reached
        elif (return_reward > self.threshold):
            self.is_finalized = True
            self.sucesses[self.total_episodes] = self.MAX_Steps

        self.traj_return += return_reward / self.current_steps
        return np.append(return_state, ((-return_reward))), return_reward, self.is_finalized, {}

    def setGolden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def setPlane(self, plane):
        if (plane == Plane.vertical or plane == Plane.horizontal):
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def seed(self, seed):
        np.random.seed(seed)

    def _take_action(self, kicks_scaled):
        dkicks = self.kicks_0 + kicks_scaled
        self.kicks_0 = dkicks.copy()
        return self._get_trajectory(dkicks, self.plane)

    def _get_trajectory(self, dkicks, plane):
        if (plane == Plane.horizontal):
            rmatrix = self.responseH
            golden = self.goldenH
        elif (plane == Plane.vertical):
            rmatrix = self.responseV
            golden = self.goldenV
        trajectory = self._calculate_trajectory(rmatrix, dkicks)
        return trajectory

    # def _calculate_response(self, bpmsTwiss, correctorsTwiss):
    #     bpms = bpmsTwiss.elements
    #     correctors = correctorsTwiss.elements
    #     rmatrix = np.zeros((len(bpms), len(correctors)))
    #     for i, bpm in enumerate(bpms):
    #         for j, corrector in enumerate(correctors):
    #             if (bpm.mu > corrector.mu):
    #                 rmatrix[i][j] = np.sqrt(bpm.beta * corrector.beta) * np.sin(
    #                     (bpm.mu - corrector.mu) * 2. * np.pi)
    #             else:
    #                 rmatrix[i][j] = 0.0
    #     return rmatrix

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        # bpms.pop(0)
        # correctors.pop(-1)

        rmatrix = np.zeros((len(bpms), len(correctors)))

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if (plane == Plane.horizontal):
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
                        (bpm_mu - corrector_mu) * 2. * np.pi)
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


if __name__ == '__main__':

    env = e_trajectory_simENV()

    environment_instance = RecordStatisticsWrapper(env)
    # environment_instance = env

    environment_instance.reset()

    actions_history = [0]


    def objective(action):
        last_action = actions_history[-1]
        action_set = action.copy() - last_action
        s, r, d, _ = environment_instance.step(action=action_set)
        actions_history.append(action.copy())
        return -r


    if True:
        environment_instance.reset()
        d = False
        steps = 0
        while steps < 5:
            d = False
            print(environment_instance.reset())

            while not d:
                sample_action = environment_instance.action_space.sample() * 0.1
                s, r, d, i = environment_instance.step(sample_action)
                print(s, sample_action)
                steps += 1
            print("episode", i)

    if False:

        def constr(action):
            if any(action > environment_instance.action_space.high[0]) or any(
                    action < environment_instance.action_space.low[0]):
                return -1
            else:
                return 1


        print('init: ', environment_instance.reset(no_limit=True))
        start_vector = np.zeros(environment_instance.action_space.shape[0])
        rhobeg = 0.25 * environment_instance.action_space.high[0]
        print('rhobeg: ', rhobeg)
        res = opt.fmin_cobyla(objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.05)
        print(res)

    if False:
        # Bounded region of parameter space
        pbounds = dict([('x' + str(i), (environment_instance.action_space.low[0],
                                        environment_instance.action_space.high[0])) for i in range(1, 12)])


        def black_box_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
            func_val = -1 * objective(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]))
            return func_val


        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=3, )

        optimizer.maximize(
            init_points=25,
            n_iter=50,
            acq="ucb",
            kappa=0.1
        )
        objective(np.array([optimizer.max['params'][x] for x in optimizer.max['params']]))

    history = np.array(environment_instance.current_history)
    rews = np.stack(history[:, -1])
    actions = np.stack(history[:, 1])
    states = np.stack(history[:, 0])

    fig, axs = plt.subplots(3, sharex=True)
    axs[2].plot(rews)
    axs[2].set_title('reward')

    axs[2].axhline(environment_instance.threshold, c='lime', ls=":")

    pd.DataFrame(actions).plot(ax=axs[0], legend=False)
    axs[0].set_title('actions')
    pd.DataFrame(states).plot(ax=axs[1], legend=False)
    axs[1].set_title('states')
    axs[1].axhline(1, c='r', ls=":")
    axs[1].axhline(-1, c='r', ls=":")
    fig.suptitle('Last episode')
    plt.tight_layout()

    plt.show()

    if len(environment_instance.return_queue) > 0:
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('episodes')
        ax1.set_ylabel('return', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(environment_instance.return_queue)

        ax2 = ax1.twinx()
        color = 'lime'
        ax2.set_ylabel('episode length', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(environment_instance.length_queue, c=color)
        # ax1.axhline(-0.1, c='red')

        # plt.title('lengths')
        fig.suptitle('Metrics')
        plt.tight_layout()
        plt.show()
