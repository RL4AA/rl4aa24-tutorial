import time

import yaml

from awake import SmartEpisodeTrackerWithPlottingWrapper, e_trajectory_simENV
from control_objects.gp_mpc_controller import GpMpcController
from utils.utils import close_run, init_control, init_visu_and_folders


def init_graphics_and_controller(env, num_steps, params_controller_dict):
    live_plot_obj = init_visu_and_folders(
        env=env, num_steps=num_steps, params_controller_dict=params_controller_dict
    )

    ctrl_obj = GpMpcController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
    )

    return live_plot_obj, ctrl_obj


def main():
    with open("config/config.yaml", "r") as file:
        params_controller_dict = yaml.safe_load(file)

    num_steps = params_controller_dict["num_steps_env"]
    num_repeat_actions = params_controller_dict["controller"]["num_repeat_actions"]
    random_actions_init = params_controller_dict["random_actions_init"]

    env = SmartEpisodeTrackerWithPlottingWrapper(
        e_trajectory_simENV()
    )
    live_plot_obj, ctrl_obj = init_graphics_and_controller(
        env, num_steps, params_controller_dict
    )

    (
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
    ) = init_control(
        ctrl_obj=ctrl_obj,
        env=env,
        live_plot_obj=live_plot_obj,
        random_actions_init=random_actions_init,
        num_repeat_actions=num_repeat_actions,
    )

    info_dict = None
    done = False
    for iter_ctrl in range(random_actions_init, num_steps):
        time_start = time.time()
        if iter_ctrl % num_repeat_actions == 0:
            if info_dict is not None:
                predicted_state = info_dict["predicted states"][0]
                predicted_state_std = info_dict["predicted states std"][0]
                check_storage = True
            else:
                predicted_state = None
                predicted_state_std = None
                check_storage = False
            # If num_repeat_actions != 1, the gaussian process models predict that much step ahead,
            # For iteration k, the memory holds obs(k - step), action (k - step), obs(k), reward(k)
            # Add memory is put before compute action because it uses data from step before
            ctrl_obj.add_memory(
                obs=obs_prev_ctrl,
                action=action,
                obs_new=obs,
                reward=-cost,
                check_storage=check_storage,
                predicted_state=predicted_state,
                predicted_state_std=predicted_state_std,
            )
            if done:
                obs, _ = env.reset()
            # Compute the action
            action, info_dict = ctrl_obj.compute_action(obs_mu=obs)
            # if params_controller_dict["verbose"]:
            if True:
                for key in info_dict:
                    print(key + ": " + str(info_dict[key]))

        # perform action on the system
        obs_new, reward, done, _, _ = env.step(action)
        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        # costs_runs[idx_test, iter_ctrl] = cost
        try:
            if live_plot_obj is not None:
                live_plot_obj.update(
                    obs=obs, cost=cost, action=action, info_dict=info_dict
                )
        except:
            print("problem in plot")
        # set obs to previous control
        obs_prev_ctrl = obs
        obs = obs_new
        print("time loop: " + str(time.time() - time_start) + " s\n")

        close_run(ctrl_obj=ctrl_obj, env=env)


if __name__ == "__main__":
    main()
