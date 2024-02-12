# torch.multiprocessing.set_sharing_strategy('file_system')
import os
import pickle
import shutil

import numpy as np
import torch
import yaml
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.envs.awake_steering_simulated import AwakeSteering as awake_env
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_input_size, get_policy_for_env
from maml_rl.utils.reinforcement_learning import get_episode_lengths
from policy_test import _layout_verficication_plot, verify


def save_progress(file_name, data, save_progress_data_dir):
    full_path = os.path.join(save_progress_data_dir, file_name)
    with open(full_path, "wb") as file:
        pickle.dump(data, file)


def main(args):
    with open(args.config, "r") as f:
        # Load the configuration from the YAML file
        config = yaml.safe_load(f)  # using safe_load for better security

    # Prepare the output folder
    base_folder = args.base_folder
    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    experiment_folder = os.path.join(base_folder, experiment_name)

    save_progress_data_dir = os.path.join(
        experiment_folder, experiment_type, "progress"
    )

    # Todo: check if that makes sense at all
    continue_fine_tuning = False  # only if one policy!

    logging_path = os.path.join(experiment_folder, experiment_type)

    if not continue_fine_tuning:
        # Remove the directory if it exists and contains files
        if os.path.exists(logging_path) and os.listdir(logging_path):
            shutil.rmtree(logging_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    time_logging_location = f"{logging_path}/total_time.csv"

    # Remove the file if it exists
    if os.path.exists(time_logging_location):
        os.remove(time_logging_location)

    if not os.path.exists(save_progress_data_dir):
        os.makedirs(save_progress_data_dir)

    # Load the predefined task
    with open(args.evaluation_tasks, "rb") as input_file:
        tasks = pickle.load(input_file)
    tasks = [tasks[id] for id in args.task_ids]
    n_tasks = len(tasks)

    # Update the config with command line arguments
    # Create a nested 'env-kwargs' dict if it does not exist
    env_kwargs = config.get("env-kwargs", {})
    # we switch to the no train mode to ensure the predefined task is read
    env_kwargs["train"] = False
    config["env-kwargs"] = env_kwargs
    config["experiment_folder"] = experiment_folder

    # Update the main config with any other command line arguments
    config.update(vars(args))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = awake_env()
    env.close()

    # Policy
    policy = get_policy_for_env(
        env, hidden_sizes=config["hidden-sizes"], nonlinearity=config["nonlinearity"]
    )

    meta_policy_location = None
    if config["use_meta_policy"] and os.path.exists(args.policy):
        with open(args.policy, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device(args.device))
            policy.load_state_dict(state_dict)
        use_task_policy = logging_path
        meta_policy_location = args.policy
    else:
        use_task_policy = logging_path
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(
        config["env-name"],
        env_kwargs=config["env-kwargs"],
        experiment_folder=config["experiment_folder"],
        experiment_type=config["experiment_type"],
        batch_size=config["fast-batch-size"],
        policy=policy,
        baseline=baseline,
        env=env,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    nr_total_interactions = 0

    # Create plot window
    fig, axes = _layout_verficication_plot(n_tasks)

    for batch in trange(args.num_batches):
        train_episodes, valid_episodes = sampler.sample(
            tasks,
            num_steps=config["num-steps"],
            fast_lr=config["fast-lr"],
            gamma=config["gamma"],
            gae_lambda=config["gae-lambda"],
            device=args.device,
        )

        save_progress(
            f"training_progress_{batch:07d}.pkl",
            [train_episodes[0], valid_episodes],
            save_progress_data_dir,
        )

        train_episodes_len = np.reshape(get_episode_lengths(train_episodes[0]), -1)
        nr_total_interactions += np.sum(train_episodes_len, axis=0)

        if ((batch + 1) % args.plot_interval) == 0:
            verify(
                tasks=tasks,
                episodes=args.num_episodes,
                use_task_policy=use_task_policy,
                meta_policy_location=meta_policy_location,
                title=f"Total interactions: {nr_total_interactions}",
                fig=fig,
                ax=axes,
                show_success_rate=True,
            )

    input("Press any key to continue.")  # Keep the plot window open


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Reinforcement learning with "
        "Model-Agnostic Meta-Learning (MAML) - Test"
    )

    parser.add_argument(
        "--config",
        type=str,
        # required=True,
        default="configs/awake_evaluation.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--policy",
        type=str,
        # required=True,
        default="awake/policy.th",
        help="path to the policy checkpoint",
    )
    parser.add_argument(
        "--use-meta-policy",
        action="store_true",
        help="use the pre-trained meta-policy",
    )
    parser.add_argument(
        "--evaluation-tasks",
        type=str,
        default="configs/evaluation_tasks.pkl",
        help="path to the evaluation tasks",
    )
    parser.add_argument(
        "--task-ids", nargs="*", type=int, default=[0], help="task ids to evaluate on"
    )
    # Output arguments
    parser.add_argument(
        "--base-folder",
        type=str,
        default="awake",
        help="path to the base folder",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="test_me",
        help="name of the experiment",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="test",
        help="experiment type",
    )

    # Evaluation
    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--num-batches", type=int, default=20, help="number of batches (default: 20)"
    )
    evaluation.add_argument(
        "--num-episodes", type=int, default=10, help="number of episodes (default: 10)"
    )

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")

    misc.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    misc.add_argument(
        "--num-workers",
        type=int,
        default=1,  # mp.cpu_count(),
        help="number of workers for trajectories sampling (default: "
        "{0})".format(mp.cpu_count()),
    )
    misc.add_argument(
        "--use-cuda",
        action="store_true",
        help="use cuda (default: false, use cpu). WARNING: Full support for cuda "
        "is not guaranteed. Using CPU is encouraged.",
    )
    misc.add_argument(
        "--plot-interval",
        type=int,
        default=1,
        help="plot the results every n batches (default: 1)",
    )

    args = parser.parse_args()
    args.device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"

    main(args)
