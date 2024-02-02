import json
import os
import pickle
import shutil
from pathlib import Path

import torch
import yaml
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.envs.awake_steering_simulated import AwakeSteering as awake_env
from maml_rl.metalearners import MAMLTRPO
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_input_size, get_policy_for_env
from maml_rl.utils.reinforcement_learning import get_returns


def prepare_folders(base_folder, experiment_type, experiment_name):
    experiment_folder = os.path.join(base_folder, experiment_name)

    save_progress_data_dir = os.path.join(
        experiment_folder, experiment_type, "progress"
    )

    training_data_location = os.path.join(experiment_folder, experiment_type)
    meta_policy_location = os.path.join(base_folder, "policy")

    # Remove the directory if it exists and has contents
    if os.path.exists(training_data_location) and os.listdir(training_data_location):
        shutil.rmtree(training_data_location)

    # Create the directory (does nothing if it already exists)
    os.makedirs(training_data_location, exist_ok=True)
    os.makedirs(save_progress_data_dir, exist_ok=True)

    # Remove the policy file if it already exists
    if os.path.exists(meta_policy_location):
        os.remove(meta_policy_location)

    time_logging_location = f"{experiment_folder}/total_time.csv"
    # Remove the file if it exists
    if os.path.exists(time_logging_location):
        os.remove(time_logging_location)


def save_progress(file_name, data, progress_data_dir):
    full_path = os.path.join(progress_data_dir, file_name)
    with open(full_path, "wb") as file:
        pickle.dump(data, file)


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_folder_path = Path(args.output_folder)
    base_folder = args.output_folder
    prepare_folders(base_folder, args.experiment_type, args.experiment_name)
    experiment_folder = os.path.join(base_folder, args.experiment_name)
    config["experiment_folder"] = experiment_folder
    save_progress_data_dir = os.path.join(
        experiment_folder, args.experiment_type, "progress"
    )
    policy_logging_dir = (
        output_folder_path / args.experiment_name / "meta_policy_logging"
    )
    policy_logging_dir.mkdir(parents=True, exist_ok=True)
    # Update the main config with any other command line arguments
    config.update(vars(args))

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        config_filename = os.path.join(args.output_folder, "config_fixed.json")

        with open(config_filename, "w") as f:
            dict_config = vars(args)
            # dict_config["env-kwargs"] = {}
            # dict_config["env-kwargs"]["train"] = False
            config.update(dict_config)
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Environment
    env_kwargs = config.get("env-kwargs", {})

    # env = DegWrapper(awake_env(**config.get("env-kwargs", {})))
    env = awake_env(**env_kwargs)
    print("env.train", env.train)
    env.close()

    # Policy
    policy = get_policy_for_env(
        env, hidden_sizes=config["hidden-sizes"], nonlinearity=config["nonlinearity"]
    )
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(
        config["env-name"],
        env_kwargs=env_kwargs,
        experiment_folder=config["experiment_folder"],
        experiment_type=config["experiment_type"],
        batch_size=config["fast-batch-size"],
        policy=policy,
        baseline=baseline,
        env=env,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    metalearner = MAMLTRPO(
        policy,
        fast_lr=config["fast-lr"],
        first_order=config["first-order"],
        device=args.device,
    )

    num_iterations = 0

    print("Starting training...")
    for batch in trange(config["num-batches"]):
        tasks = sampler.sample_tasks(num_tasks=config["meta-batch-size"])
        futures = sampler.sample_async(
            tasks,
            num_steps=config["num-steps"],
            fast_lr=config["fast-lr"],
            gamma=config["gamma"],
            gae_lambda=config["gae-lambda"],
            device=args.device,
        )
        logs = metalearner.step(
            *futures,
            max_kl=config["max-kl"],
            cg_iters=config["cg-iters"],
            cg_damping=config["cg-damping"],
            ls_max_steps=config["ls-max-steps"],
            ls_backtrack_ratio=config["ls-backtrack-ratio"],
        )

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(
            tasks=tasks,
            num_iterations=num_iterations,
            train_returns=get_returns(train_episodes[0]),
            valid_returns=get_returns(valid_episodes),
        )

        # Save progress
        if (batch + 1) % args.save_progress_interval == 0:
            # Log Training Progress
            save_progress(
                f"training_progress_{batch: 07d}.pkl",
                [train_episodes[0], valid_episodes],
                save_progress_data_dir,
            )
            # Save meta-policy
            policy_name_current = f"meta_policy_{batch:07d}.th"  # Format batch number
            torch.save(policy.state_dict(), policy_logging_dir / policy_name_current)

            # Save a global meta policy
            torch.save(policy.state_dict(), output_folder_path / "policy.th")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Reinforcement learning with "
        "Model-Agnostic Meta-Learning (MAML) - Train on AWAKE"
    )

    parser.add_argument(
        "--config",
        type=str,
        # required=True,
        default="configs/awake.yaml",
        help="path to the configuration file.",
    )

    # Output
    output = parser.add_argument_group("Miscellaneous")
    output.add_argument(
        "--output-folder", type=str, default="awake", help="name of the output folder"
    )
    output.add_argument(
        "--experiment-name", type=str, default="test_me", help="name of the experiment"
    )
    output.add_argument(
        "--experiment-type", type=str, default="train", help="experiment type"
    )

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=int, default=None, help="random seed")
    misc.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
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
        "--save-progress-interval",
        type=int,
        default=1,
        help="interval between saving progress",
    )

    args = parser.parse_args()
    args.device = "cpu"
    torch.multiprocessing.set_start_method("spawn")
    print(args)
    main(args)
