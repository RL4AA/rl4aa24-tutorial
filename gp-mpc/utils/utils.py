import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from control_objects.gp_models import ExactGPModelMonoTask

matplotlib.rc("font", size="6")
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6

# Values used for graph visualizations only
MEAN_PRED_COST_INIT = 0
STD_MEAN_PRED_COST_INIT = 1

N = 10

colors_map = cm.rainbow(np.linspace(0, 1, N))


def create_models(
    train_inputs,
    train_targets,
    params,
    constraints_gp,
    num_models=None,
    num_inputs=None,
):
    """
    Define gaussian process models used for predicting state transition,
    using constraints and init values for (outputscale, noise, lengthscale).

    Args:
            train_inputs (torch.Tensor or None): Input values in the memory of the gps
            train_targets (torch.Tensor or None): target values in the memory of the gps.
                                                                                            Represent the change in state values
            params (dict or list of dict): Value of the hyper-parameters of the gaussian processes.
                                                                            Dict type is used to create the models with init values from the json file.
                                                                            List of dict type is used to create the models
                                                                                                    with exported parameters, such as in the parallel training process.
            constraints_gp (dict): See the ReadMe about parameters for information about keys
            num_models (int or None): Must be provided when train_inputs or train_targets are None.
                                                                    The number of models should be equal to the dimension of state,
                                                                    so that the transition for each state can be predicted with a different gp.
                                                                    Default=None
            num_inputs (int or None): Must be provided when train_inputs or train_targets are None.
                                                                    The number of inputs should be equal to the sum of the dimension of state
                                                                    and dimension of action. Default=None
            include_time (bool): If True, gp will have one additional input corresponding to the time of the observation.
                                                            This is usefull if the env change with time,
                                                            as more recent points will be trusted more than past points
                                                            (time closer to the point to make inference at).
                                                            It is to be specified only if

    Returns:
            models (list of gpytorch.models.ExactGP): models containing the parameters, memory,
                                                                                                    constraints of the gps and functions for exact predictions
    """
    if train_inputs is not None and train_targets is not None:
        num_models = len(train_targets[0])
        models = [
            ExactGPModelMonoTask(
                train_inputs, train_targets[:, idx_model], len(train_inputs[0])
            )
            for idx_model in range(num_models)
        ]
    else:
        if num_models is None or num_inputs is None:
            raise (
                ValueError(
                    "If train_inputs or train_targets are None, num_models and num_inputs must be defined"
                )
            )
        else:
            models = [
                ExactGPModelMonoTask(None, None, num_inputs) for _ in range(num_models)
            ]

    for idx_model in range(num_models):
        if constraints_gp is not None:
            if "min_std_noise" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_std_noise"]) != float
                    and type(constraints_gp["min_std_noise"]) != int
                ):
                    min_var_noise = np.power(
                        constraints_gp["min_std_noise"][idx_model], 2
                    )
                else:
                    min_var_noise = np.power(constraints_gp["min_std_noise"], 2)
                if (
                    type(constraints_gp["max_std_noise"]) != float
                    and type(constraints_gp["max_std_noise"]) != int
                ):
                    max_var_noise = np.power(
                        constraints_gp["max_std_noise"][idx_model], 2
                    )
                else:
                    max_var_noise = np.power(constraints_gp["max_std_noise"], 2)
                models[idx_model].likelihood.noise_covar.register_constraint(
                    "raw_noise",
                    gpytorch.constraints.Interval(
                        lower_bound=min_var_noise, upper_bound=max_var_noise
                    ),
                )

            if "min_outputscale" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_outputscale"]) != float
                    and type(constraints_gp["min_outputscale"]) != int
                ):
                    min_outputscale = constraints_gp["min_outputscale"][idx_model]
                else:
                    min_outputscale = constraints_gp["min_outputscale"]

                if (
                    type(constraints_gp["max_outputscale"]) != float
                    and type(constraints_gp["max_outputscale"]) != int
                ):
                    max_outputscale = constraints_gp["max_outputscale"][idx_model]
                else:
                    max_outputscale = constraints_gp["max_outputscale"]
                models[idx_model].covar_module.register_constraint(
                    "raw_outputscale",
                    gpytorch.constraints.Interval(
                        lower_bound=min_outputscale, upper_bound=max_outputscale
                    ),
                )

            if "min_lengthscale" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_lengthscale"]) == float
                    or type(constraints_gp["min_lengthscale"]) == int
                ):
                    min_lengthscale = constraints_gp["min_lengthscale"]
                else:
                    min_lengthscale = constraints_gp["min_lengthscale"][idx_model]
                if (
                    type(constraints_gp["max_lengthscale"]) == float
                    or type(constraints_gp["max_lengthscale"]) == int
                ):
                    max_lengthscale = constraints_gp["max_lengthscale"]
                else:
                    max_lengthscale = constraints_gp["max_lengthscale"][idx_model]

                models[idx_model].covar_module.base_kernel.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.Interval(
                        lower_bound=min_lengthscale, upper_bound=max_lengthscale
                    ),
                )

        # load parameters
        # dict type is used when initializing the models from the json config file
        # list type is used when initializing the models in the parallel training process
        # using the exported parameters
        if type(params) == dict:
            hypers = {
                "base_kernel.lengthscale": params["base_kernel.lengthscale"][idx_model],
                "outputscale": params["outputscale"][idx_model],
            }
            hypers_likelihood = {
                "noise_covar.noise": params["noise_covar.noise"][idx_model]
            }
            models[idx_model].likelihood.initialize(**hypers_likelihood)
            models[idx_model].covar_module.initialize(**hypers)
        elif type(params) == list:
            models[idx_model].load_state_dict(params[idx_model])
    return models


def init_control(
    ctrl_obj, env, live_plot_obj, random_actions_init, num_repeat_actions=1
):
    """
    Initializes the control environment with random actions and updates visualization and memory.

    Args:
        ctrl_obj (GpMpcController): Control object for computing cost and managing memory.
        env (gym.Env): Gym environment for obtaining observations and applying actions.
        live_plot_obj: Object for real-time 2D graph visualization, with an `update` method.
        rec: Real-time environment visualization object, with a `capture_frame` method.
        params_general (dict): General parameters (render_env, save_render_env, render_live_plots_2d).
        random_actions_init (int): Number of initial random actions.
        costs_tests (np.array): Array to store costs for analysis (shape: (num_runs, num_timesteps)).
        idx_test (int): Current test index.
        num_repeat_actions (int): Number of consecutive constant actions; affects memory storage.
    """
    obs_lst, actions_lst, rewards_lst = [], [], []
    obs, _ = env.reset()
    action, cost, obs_prev_ctrl = None, None, None
    done = False
    for idx_action in range(random_actions_init):
        if idx_action % num_repeat_actions == 0 or action is None:
            action = env.action_space.sample()
            if obs_prev_ctrl is not None and cost is not None:
                ctrl_obj.add_memory(
                    obs=obs_prev_ctrl,
                    action=action,
                    obs_new=obs,
                    reward=cost,
                    check_storage=False,
                )
        if done:
            obs, _ = env.reset()
        obs_new, reward, done, _, _ = env.step(action)
        obs_prev_ctrl = obs
        obs = obs_new
        cost, _ = ctrl_obj.compute_cost_unnormalized(obs_new, action)
        # cost=reward!
        cost = cost

        # Update lists for visualization
        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(cost)

        # if params_general.get('render_live_plots_2d'):
        live_plot_obj.update(obs=obs, action=action, cost=cost, info_dict=None)

        # Store the last action for potential future use
        ctrl_obj.action_previous_iter = (
            action  # Adjust as needed, e.g., convert to tensor
        )

    return (
        ctrl_obj,
        env,
        live_plot_obj,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        obs_lst,
        actions_lst,
        rewards_lst,
    )


def init_visu_and_folders(env, num_steps, params_controller_dict):
    """
    Create and return the objects used for visualisation in real-time.
    Also create the folder where to save the figures.
    Args:
            env (gym env): gym environment
            num_steps (int): number of steps during which the action is maintained constant
            env_str (str): string that describes the env name
            params_general (dict): general parameters (see parameters.md for more info)
            params_controller_dict (dict): controller parameters (see parameters.md for more info)

        Returns:
                live_plot_obj (object): object used to visualize the control and observation evolution in a 2d graph
    """
    live_plot_obj = LivePlotSequential(
        num_steps,
        env.observation_space,
        env.action_space,
        step_pred=params_controller_dict["controller"]["num_repeat_actions"],
    )

    return live_plot_obj


def close_run(ctrl_obj, env):
    """
    Close all visualisations and parallel processes that are still running.
    Save all visualisations one last time if save args set to True
    Args:
            ctrl_obj:
            env (gym env): gym environment
    """

    env.__exit__()
    ctrl_obj.check_and_close_processes()


class LivePlotSequential:
    def __init__(
        self,
        num_steps_total,
        obs_space,
        action_space,
        step_pred,
        mul_std_bounds=3,
        fontsize=6,
    ):
        self.fig, self.axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
        self.axes[0].set_title("Normed states and predictions")
        self.axes[1].set_title("Normed actions")
        self.axes[2].set_title("Reward and horizon reward")
        plt.xlabel("Env steps")
        self.min_state = -0.03
        self.max_state = 1.03
        self.axes[0].set_ylim(self.min_state, self.max_state)
        self.axes[1].set_ylim(-0.03, 1.03)
        self.axes[2].set_xlim(0, num_steps_total)
        plt.tight_layout()

        self.step_pred = step_pred
        self.mul_std_bounds = mul_std_bounds

        self.states = np.empty((num_steps_total, obs_space.shape[0]))
        self.actions = np.empty((num_steps_total, action_space.shape[0]))
        self.costs = np.empty(num_steps_total)

        self.mean_costs_pred = np.empty_like(self.costs)
        self.mean_costs_std_pred = np.empty_like(self.costs)

        self.min_obs = obs_space.low
        self.max_obs = obs_space.high
        self.min_action = action_space.low
        self.max_action = action_space.high

        self.num_points_show = 0
        self.lines_states = [
            self.axes[0].plot(
                [],
                [],
                label="state" + str(state_idx),
                color=colors_map[state_idx],  # cmap.colors[2 * state_idx]
            )
            for state_idx in range(obs_space.shape[0])
        ]
        self.line_cost = self.axes[2].plot([], [], label="reward", color="k")

        self.lines_actions = [
            self.axes[1].step(
                [],
                [],
                label="action" + str(action_idx),
                color=colors_map[action_idx],
            )
            for action_idx in range(action_space.shape[0])
        ]

        self.line_mean_costs_pred = self.axes[2].plot(
            [], [], label="mean predicted reward", color="orange"
        )
        self.lines_states_pred = [
            self.axes[0].plot(
                [],
                [],
                # [],
                label="predicted_states" + str(state_idx),
                color=colors_map[state_idx],
                linestyle="dashed",
            )
            for state_idx in range(obs_space.shape[0])
        ]
        self.lines_actions_pred = [
            self.axes[1].step(
                [],
                [],
                label="predicted_action" + str(action_idx),
                color=colors_map[action_idx],
                linestyle="dashed",
            )
            for action_idx in range(action_space.shape[0])
        ]
        self.line_costs_pred = self.axes[2].plot(
            [], [], label="predicted cost", color="k", linestyle="dashed"
        )

        self.axes[0].legend(fontsize=fontsize)
        self.axes[0].grid()
        self.axes[1].legend(fontsize=fontsize)
        self.axes[1].grid()
        self.axes[2].legend(fontsize=fontsize)
        self.axes[2].grid()
        plt.show(block=False)

    def update(self, obs, action, cost, info_dict=None):
        obs_norm = (obs - self.min_obs) / (self.max_obs - self.min_obs)
        action_norm = (action - self.min_action) / (self.max_action - self.min_action)
        self.states[self.num_points_show] = obs_norm
        self.costs[self.num_points_show] = -cost

        update_limits = False
        min_state_actual = np.min(obs_norm)
        if min_state_actual < self.min_state:
            self.min_state = min_state_actual
            update_limits = True

        max_state_actual = np.max(obs_norm)
        if max_state_actual > self.max_state:
            self.max_state = max_state_actual
            update_limits = True

        if update_limits:
            self.axes[0].set_ylim(self.min_state, self.max_state)

        idxs = np.arange(0, (self.num_points_show + 1))
        for idx_axes in range(len(self.axes)):
            # self.axes[idx_axes].collections.clear()
            # print(self.axes[idx_axes])
            # self.axes[idx_axes].clear()
            for collection in self.axes[idx_axes].collections:
                collection.remove()

        for idx_state in range(len(obs_norm)):
            self.lines_states[idx_state][0].set_data(
                idxs, self.states[: (self.num_points_show + 1), idx_state]
            )

        self.actions[self.num_points_show] = action_norm
        for idx_action in range(len(action_norm)):
            self.lines_actions[idx_action][0].set_data(
                idxs, self.actions[: (self.num_points_show + 1), idx_action]
            )

        self.line_cost[0].set_data(idxs, self.costs[: (self.num_points_show + 1)])

        if info_dict is not None:
            mean_costs_pred = info_dict["mean predicted cost"]
            mean_costs_pred *= -1
            mean_costs_std_pred = info_dict["mean predicted cost std"]
            states_pred = info_dict["predicted states"]
            states_std_pred = info_dict["predicted states std"]
            actions_pred = info_dict["predicted actions"]
            costs_pred = info_dict["predicted costs"]
            costs_pred *= -1
            costs_std_pred = info_dict["predicted costs std"]
            np.nan_to_num(mean_costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(mean_costs_std_pred, copy=False, nan=99, posinf=99, neginf=99)
            np.nan_to_num(states_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(states_std_pred, copy=False, nan=99, posinf=99, neginf=0)
            np.nan_to_num(costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(costs_std_pred, copy=False, nan=99, posinf=99, neginf=0)
            # if num_repeat_action is not 1, the control do not happen at each iteration,
            # we must select the last index where the control happened as the start of the prediction horizon
            idx_prev_control = (idxs[-1] // self.step_pred) * self.step_pred
            idxs_future = np.arange(
                idx_prev_control,
                idx_prev_control + self.step_pred + len(states_pred) * self.step_pred,
                self.step_pred,
            )

            self.mean_costs_pred[self.num_points_show] = mean_costs_pred
            self.mean_costs_std_pred[self.num_points_show] = mean_costs_std_pred

            for idx_state in range(len(obs_norm)):
                future_states_show = np.concatenate(
                    (
                        [self.states[idx_prev_control, idx_state]],
                        states_pred[:, idx_state],
                    )
                )
                self.lines_states_pred[idx_state][0].set_data(
                    idxs_future, future_states_show
                )
                future_states_std_show = np.concatenate(
                    ([0], states_std_pred[:, idx_state])
                )
                self.axes[0].fill_between(
                    idxs_future,
                    future_states_show - future_states_std_show * self.mul_std_bounds,
                    future_states_show + future_states_std_show * self.mul_std_bounds,
                    facecolor=colors_map[idx_state],
                    alpha=ALPHA_CONFIDENCE_BOUNDS,
                    label="predicted "
                    + str(self.mul_std_bounds)
                    + " std bounds state "
                    + str(idx_state),
                )
            for idx_action in range(len(action_norm)):
                self.lines_actions_pred[idx_action][0].set_data(
                    idxs_future,
                    np.concatenate(
                        (
                            [self.actions[idx_prev_control, idx_action]],
                            actions_pred[:, idx_action],
                        )
                    ),
                )

            future_costs_show = np.concatenate(
                ([self.costs[idx_prev_control]], costs_pred)
            )
            self.line_costs_pred[0].set_data(idxs_future, future_costs_show)

            future_cost_std_show = np.concatenate(([0], costs_std_pred))
            self.axes[2].fill_between(
                idxs_future,
                future_costs_show - future_cost_std_show * self.mul_std_bounds,
                future_costs_show + future_cost_std_show * self.mul_std_bounds,
                facecolor="black",
                alpha=ALPHA_CONFIDENCE_BOUNDS,
                label="predicted " + str(self.mul_std_bounds) + " std cost bounds",
            )
        else:
            if self.num_points_show == 0:
                self.mean_costs_pred[self.num_points_show] = MEAN_PRED_COST_INIT
                self.mean_costs_std_pred[self.num_points_show] = STD_MEAN_PRED_COST_INIT
            else:
                self.mean_costs_pred[self.num_points_show] = self.mean_costs_pred[
                    self.num_points_show - 1
                ]
                self.mean_costs_std_pred[self.num_points_show] = (
                    self.mean_costs_std_pred[self.num_points_show - 1]
                )

        self.line_mean_costs_pred[0].set_data(
            idxs, self.mean_costs_pred[: (self.num_points_show + 1)]
        )
        self.axes[2].set_ylim(
            np.min(
                [
                    np.min(self.mean_costs_pred[: (self.num_points_show + 1)]),
                    np.min(self.costs[: (self.num_points_show + 1)]),
                ]
            )
            * 1.1,
            0.5,
        )
        self.axes[2].fill_between(
            idxs,
            self.mean_costs_pred[: (self.num_points_show + 1)]
            - self.mean_costs_std_pred[: (self.num_points_show + 1)]
            * self.mul_std_bounds,
            self.mean_costs_pred[: (self.num_points_show + 1)]
            + self.mean_costs_std_pred[: (self.num_points_show + 1)]
            * self.mul_std_bounds,
            facecolor="orange",
            alpha=ALPHA_CONFIDENCE_BOUNDS,
            label="mean predicted " + str(self.mul_std_bounds) + " std cost bounds",
        )

        self.fig.canvas.draw()
        plt.pause(0.01)
        self.num_points_show += 1
