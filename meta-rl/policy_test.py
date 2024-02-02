import json
import os
import pickle
from functools import reduce
from operator import mul
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from maml_rl.envs.awake_steering_simulated import AwakeSteering as awake_env
from maml_rl.envs.helpers import Awake_Benchmarking_Wrapper
from maml_rl.utils.helpers import get_policy_for_env

# get default color cycle
plt_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_input_size(env):
    return reduce(mul, env.observation_space.shape, 1)


def ep_to_list(data_per_ep, num_episodes_to_show=5):
    """Rollout the data into a single list"""
    max_episode = min(len(data_per_ep), num_episodes_to_show)
    flattened_list = []
    for ep_data in data_per_ep[:max_episode]:
        for data in ep_data:
            flattened_list.append(data)
    return flattened_list


def ep_mean_return(rewards_per_task):
    mean_returns = np.mean([np.sum(rews) for rews in rewards_per_task])
    return mean_returns


def ep_success_rate(rewards_per_task):
    success_rate = np.mean([rews[-1] > -1 for rews in rewards_per_task])
    return success_rate


def plot_episode_lengths(ax_len, ax_cum_len, ep_len_per_task, ep_len_opt_per_task):
    for i, ep_len in enumerate(ep_len_per_task):
        x = np.arange(len(ep_len))
        ax_len.plot(x, ep_len, label=f"Policy Task {i}", color=plt_colors[i])
        ax_len.plot(
            x,
            ep_len_opt_per_task[i],
            label="Benchmark Policy",
            ls="--",
            color=plt_colors[i],
        )

        ax_cum_len.plot(x, np.cumsum(ep_len), color=plt_colors[i])
        ax_cum_len.plot(
            x, np.cumsum(ep_len_opt_per_task[i]), ls="--", color=plt_colors[i]
        )
    ax_len.set_ylabel("Episode length")
    ax_cum_len.set_ylabel("Cumulative \n episode length")
    ax_cum_len.set_xlabel("Episode")
    ax_len.set_xticks([])
    return ax_len, ax_cum_len


def plot_actions_states(
    ax, data_per_task, label, show_x=True, title=None, num_episodes_to_show=5, i_task=0
):
    data_per_ep = data_per_task[i_task]  # Only plot one task
    max_episode = min(len(data_per_ep), num_episodes_to_show)
    x_start = 0
    for episode in range(max_episode):
        ax.set_prop_cycle(None)
        data_list = data_per_ep[episode]
        x = np.linspace(x_start, x_start + len(data_list), len(data_list))
        ax.plot(
            x,
            data_list,
            label=f"{label} Task {i_task}",
            marker=".",
        )
        x_start += len(data_list) + 1

    # data_list = ep_to_list(data, num_episodes_to_show)
    # x = np.linspace(0, len(data_list), len(data_list))
    # ax.plot(x, data_list, label=f"{label} Task {i_task}")
    ax.set_ylabel(label)

    if show_x:
        ax.set_xlabel("Step")
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title)


def plot_rewards(
    ax_cum_reward,
    ax_final_reward,
    rewards_per_task,
    rewards_optimal_per_task,
):
    for i, rews in enumerate(rewards_per_task):
        ep_returns = [np.sum(r) for r in rews]
        x = np.linspace(0, len(ep_returns), len(ep_returns))
        ax_cum_reward.plot(x, ep_returns, label=f"Policy Task {i}", color=plt_colors[i])
        ax_cum_reward.plot(
            x,
            rewards_optimal_per_task[i],
            label="Benchmark Policy",
            ls="--",
            color=plt_colors[i],
        )

        ep_final_rews = [r[-1] for r in rews]
        ax_final_reward.plot(
            x,
            ep_final_rews,
            label=f"Len Task {i}",
            color=plt_colors[i],
            marker="o",
            ls="",
        )
        ax_final_reward.axhline(-1, ls="-", color="red")
    ax_cum_reward.set_ylabel("Cumulative reward")
    ax_final_reward.set_ylabel("Final reward")
    ax_cum_reward.set_xticks([])
    ax_final_reward.set_xlabel("Episode")
    return ax_cum_reward, ax_final_reward


def _layout_verficication_plot(n_tasks=1, figsize=(10, 6), task_ids=None, **kwargs):
    # Create figure and axes, fix layout beforehand
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(4, 2, height_ratios=[1, 1, 0.5, 1])
    fig.subplots_adjust(
        hspace=0.05, wspace=0.2, top=0.88, bottom=0.1, left=0.1, right=0.95
    )
    # Invisible axes for labels
    axes[2, 0].set_visible(False)
    axes[2, 1].set_visible(False)
    legend_policy = [
        Line2D([0], [0], color="black", lw=2, ls="-", label="RL Policy"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="Benchmark Policy"),
        Line2D([0], [0], color="red", lw=2, ls="-", label="Threshold"),
    ]
    legend_tasks = []
    if task_ids is None:
        task_ids = list(range(n_tasks))
    else:
        assert len(task_ids) == n_tasks
    for i, task_id in enumerate(task_ids):
        legend_tasks.append(
            Patch(
                facecolor=plt_colors[i],
                label=f"Task {task_id}",
            )
        )
    fig.legend(handles=legend_policy, loc="upper left", ncol=2)
    fig.legend(handles=legend_tasks, loc="upper right", ncol=min(4, n_tasks))

    return fig, axes.flatten()[[0, 1, 2, 3, 6, 7]]


def plot_verification(
    rewards_per_task,
    ep_len_per_task,
    actions_per_task,
    states_per_task,
    optimal_data_per_task,
    show_plot=True,
    **kwargs,
):
    # Check if fig and ax are passed as kwargs
    fig = kwargs.get("fig")
    axes = kwargs.get("ax")
    # Axes in form [ep_len, rewards, ep_len_cum, rewards_final, actions, states]
    if fig is None or axes is None:
        # Create figure and axes
        fig, axes = _layout_verficication_plot(n_tasks=len(ep_len_per_task), **kwargs)
    else:
        # Clear axes
        for a in axes:
            a.clear()

    # Pre-compute these values
    ep_len_optimal_per_task = []
    rewards_optimal_per_task = []
    for optimal_data in optimal_data_per_task:
        ep_len_opt = [len(data[1]) - 1 for data in optimal_data]
        rewards_opt = [np.sum(data[2][1:-1]) for data in optimal_data]
        ep_len_optimal_per_task.append(ep_len_opt)
        rewards_optimal_per_task.append(rewards_opt)

    # Plot episode lengths
    plot_episode_lengths(axes[0], axes[2], ep_len_per_task, ep_len_optimal_per_task)

    # Plot rewards
    plot_rewards(axes[1], axes[3], rewards_per_task, rewards_optimal_per_task)

    # Plot actions and states
    num_episodes_to_show = kwargs.get("num_episodes_to_show", 5)
    plot_actions_states(
        axes[4],
        actions_per_task,
        label="Actions",
        title="Steering angles",
        num_episodes_to_show=num_episodes_to_show,
    )
    plot_actions_states(
        axes[5],
        states_per_task,
        "States",
        title="BPM readings",
        num_episodes_to_show=num_episodes_to_show,
    )

    ep_break_points = np.cumsum(ep_len_per_task[0][:num_episodes_to_show])
    ep_break_points_actions = ep_break_points + np.arange(1, num_episodes_to_show + 1)
    ep_break_points_states = (
        ep_break_points + np.arange(1, num_episodes_to_show + 1) * 2
    )
    for i in range(num_episodes_to_show):
        axes[4].axvline(ep_break_points_actions[i] - 0.5, color="grey", ls="--", lw=2)
        # Fill in grey box for episode breakpoints

        # axes[5].axvspan(
        #     ep_break_points_states[i] - 0.5,
        #     ep_break_points_states[i] + 0.5,
        #     facecolor="grey",
        #     alpha=0.5,
        # )
        axes[5].axvline(ep_break_points_states[i] - 0.5, color="grey", ls="--", lw=2)

    # Handle additional kwargs
    title = kwargs.get("title")
    if title:
        fig.suptitle(title)

    # Show sucess rate
    if kwargs.get("show_success_rate"):
        success_rate_str = "Success rate "
        for i, success_rate_per_task in enumerate(
            map(ep_success_rate, rewards_per_task)
        ):
            success_rate_str += f"Task {i}: {success_rate_per_task*100:.1f}%  "
        # hack for the position
        axes[0].text(
            1.0,
            1.1,
            success_rate_str,
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # Align y-labels
    fig.align_ylabels(axes[:4])
    fig.align_ylabels(axes[4:])

    # Save plot
    save_folder = kwargs.get("save_folder")
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        fig.savefig(f"{save_folder}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_folder}.png", bbox_inches="tight", dpi=300)

    # Show plot
    if show_plot:
        plt.draw()
        plt.pause(0.5)


def load_policy_config(config_loc):
    with open(config_loc, "r") as f:
        config = json.load(f)
    return config


def load_stored_policy(
    env,
    name=None,
    config_loc="awake/config_fixed.json",
    meta_policy_location="awake/policy.th",
):
    config = load_policy_config(config_loc)
    policy = get_policy_for_env(
        env, hidden_sizes=config["hidden-sizes"], nonlinearity=config["nonlinearity"]
    ).float()

    def policy_function(x, params=None):
        """Applies the policy to input x with optional parameters."""
        return policy(x, params=params) if params else policy(x)

    if name:
        print(name)
        policy_path = Path(meta_policy_location) if name == "meta" else Path(name)

        if policy_path.exists():
            try:
                with policy_path.open("rb") as f:
                    if name == "meta":
                        print(f"Loading meta policy from {policy_path}...")
                        state_dict = torch.load(f)
                        policy.load_state_dict(state_dict)
                    else:
                        print(f"Loading individual policy from {policy_path}...")
                        params = np.load(f, allow_pickle=True).tolist()
                        return lambda x: policy_function(x, params=params)
            except IOError as e:
                print(f"Error opening policy file {policy_path}: {e}")
            except torch.TorchError as e:
                print(f"Error loading policy state for {policy_path}: {e}")
            except np.lib.npyio.NpyioError as e:
                print(f"Error loading parameters from {policy_path}: {e}")
        else:
            print("No such file loading random init")
    else:
        print("return random init policy")
    return policy_function


def policy_act_fun(policy, observations):
    # Assuming 'observations' is already in batch form
    # and policy is on the same device (CPU or GPU)
    observations_tensor = torch.from_numpy(observations).float()
    with torch.no_grad():  # Disable gradient computation for inference
        actions_tensor = policy(observations_tensor).sample()
    actions = (
        actions_tensor.cpu().numpy()
    )  # Convert back to numpy array only if necessary
    return actions


def create_trajectories_with_policies_on_different_tasks(policies, tasks, episodes=200):
    (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_per_task,
    ) = ([], [], [], [], [])

    for num_task, task in enumerate(tasks):
        env = Awake_Benchmarking_Wrapper(awake_env(task=task, train=False))
        policy_act = lambda x, num_task=num_task: policy_act_fun(  # noqa: E731
            policies[num_task], x
        )

        (
            ep_len_list,
            state_list,
            actions_list,
            reward_list,
            optimal_data_list,
        ) = create_trajectories(env, policy_act, episodes)

        ep_len_per_task.append(ep_len_list)
        rewards_per_task.append(reward_list)
        actions_per_task.append(actions_list)
        states_per_task.append(state_list)
        optimal_per_task.append(optimal_data_list)

    for i, ep_ret in enumerate(map(ep_mean_return, rewards_per_task)):
        print(f"Mean return for Task nr.{i}: {ep_ret}")

    for i, ep_ret in enumerate(map(ep_success_rate, rewards_per_task)):
        print(f"Success rate Task nr.{i}: {ep_ret}")

    return (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_per_task,
    )


def test_policies(
    tasks=[], episodes=200, use_task_policy=False, meta_policy_location=None
):
    # with open(r"maml_rl/envs/Tasks_data/Tasks_new.pickle", "rb") as input_file:
    #     tasks = pickle.load(input_file)[:tasks]

    policies = []
    env = awake_env()

    for num_task, _ in enumerate(tasks):
        if use_task_policy == "meta":
            policy_file = use_task_policy
        elif use_task_policy:  # True but not "meta"
            policy_file = os.path.join(use_task_policy, f"policy_{num_task}.npy")
        else:
            policy_file = None

        policies.append(
            load_stored_policy(
                env, policy_file, meta_policy_location=meta_policy_location
            )
        )

    (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    ) = create_trajectories_with_policies_on_different_tasks(
        policies, tasks, episodes=episodes
    )

    return (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    )


def test_policies_ext(env, policy=None, episodes=50):
    (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    ) = ([], [], [], [], [])

    if not policy:
        policy = lambda x: env.action_space.sample()  # noqa: E731

    (
        ep_len_list,
        state_list,
        actions_list,
        reward_list,
        optimal_data_list,
    ) = create_trajectories(env, policy, episodes)

    ep_len_per_task.append(ep_len_list)
    rewards_per_task.append(reward_list)
    actions_per_task.append(actions_list)
    states_per_task.append(state_list)
    optimal_data_per_task.append(optimal_data_list)

    for i, ep_return in enumerate(map(ep_mean_return, rewards_per_task)):
        print(f"Mean return for Task nr.{i}: {ep_return}")

    for i, ep_return in enumerate(map(ep_success_rate, rewards_per_task)):
        print(f"Success rate Task nr.{i}: {ep_return}")

    return (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    )


def create_trajectories(env, policy, episodes):
    ep_len_list, observation_list, actions_list, reward_list, optimal_data_list = (
        [],
        [],
        [],
        [],
        [],
    )

    for _ in range(episodes):
        observations, actions, rewards = [], [], []
        observation, info = env.reset()
        observations.append(observation)
        optimal_data = env.get_optimal_trajectory()
        ep_len = 0

        done = False
        while not done:
            action = policy(observation)
            observation, reward, terminated, truncated, infos = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            done = terminated or truncated
            ep_len += 1

        ep_len_list.append(ep_len)
        optimal_data_list.append(optimal_data)
        actions_list.append(actions)
        observation_list.append(observations)
        reward_list.append(rewards)

    return ep_len_list, observation_list, actions_list, reward_list, optimal_data_list


def verify(tasks=[], episodes=50, use_task_policy=False, **kwargs):
    meta_policy_location = kwargs.get("meta_policy_location", None)
    # Check meta policy oon test env
    (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    ) = test_policies(
        tasks=tasks,
        episodes=episodes,
        use_task_policy=use_task_policy,
        meta_policy_location=meta_policy_location,
    )
    plot_verification(
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
        **kwargs,
    )


def verify_external_policy_on_specific_env(env, policy, episodes=50, **kwargs):
    env = Awake_Benchmarking_Wrapper(env)
    # Check meta policy oon test env
    (
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
    ) = test_policies_ext(env, policy, episodes)
    kwargs["threshold"] = env.unwrapped.threshold
    plot_verification(
        rewards_per_task,
        ep_len_per_task,
        actions_per_task,
        states_per_task,
        optimal_data_per_task,
        **kwargs,
    )


if __name__ == "__main__":
    num_tasks = 1
    episodes = 50
    task_policy = "meta"
    # load the predefined task
    # Define file location and name
    verification_tasks_loc = "configs"
    filename = "verification_tasks.pkl"  # Adding .pkl extension for clarity
    # Construct the full file path
    full_path = os.path.join(verification_tasks_loc, filename)

    with open(full_path, "rb") as input_file:  # Load in tasks
        tasks = pickle.load(input_file)

    task_selected = tasks[0]
    tasks = [task_selected]

    # training_data_location = "awake/policy.th"

    print("META POLICY")
    task_policy = "meta"
    verify(tasks=tasks, episodes=episodes, use_task_policy_directory=task_policy)
    print("Random init POLICY")
    task_policy = None
    verify(tasks=tasks, episodes=episodes, use_task_policy_directory=task_policy)
    # print('Task POLICY', training_data_location)
    # task_policy = training_data_location
    # verify(tasks=tasks, episodes=episodes, use_task_policy_directory=task_policy)
    input("Press Enter to continue...")  # To not close the plot
