import argparse
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

from maml_rl.envs.awake_steering_simulated import AwakeSteering as awake_env
from policy_test import verify_external_policy_on_specific_env


def main(args):
    # load the predefined task
    with open(args.evaluation_tasks, "rb") as f:  # Load in tasks
        tasks = pickle.load(f)

    seed = args.seed
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if args.task_id == -1:
        env = awake_env(train=False, seed=seed)
        task = env.sample_tasks(1)[0]  # Sample a single task
        env.reset_task(task)
    elif args.task_id < len(tasks) and args.task_id >= 0:
        task = tasks[args.task_id]  # Train PPO on a single task
        env = awake_env(task=task, train=False, seed=seed)
    else:
        raise ValueError(
            f"Task ID {args.task_id} is not valid."
            + f"Please provide a task ID between 0 and {len(tasks) - 1}, or -1."
        )

    if args.train:
        print("Training model...")
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=seed, tensorboard_log="./logs/ppo/"
        )
        model.set_random_seed(seed)
        if args.steps > model.n_steps:
            model.learn(total_timesteps=args.steps)
        model.save(args.output_file)
    else:
        print("Loading model...")
        model = PPO.load(args.output_file)

    def get_deterministic_policy(x):
        return model.predict(x)[0]
        # return model.action_space.sample()

    policy = get_deterministic_policy

    print(f"Verifying policy on task {args.task_id}...")
    verify_external_policy_on_specific_env(
        env,
        policy,
        tasks=[task],
        episodes=100,
        title="Evaluation Plot: PPO Agent",
        save_folder="figures/ppo_evaluation",
        task_ids=[args.task_id],
    )

    input("Press Enter to continue...")  # To not close the plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Awake Steering")
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="total number of interactions with the environment",
    )
    parser.add_argument(
        "--evaluation-tasks",
        type=str,
        default="configs/evaluation_tasks.pkl",
        help="path to the evaluation tasks",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="ppo_awake",
        help="Name of the model output",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save it.",
    )
    parser.add_argument(
        "--test",
        dest="train",
        action="store_false",
        help="Only load the model and evaluate it.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID to train/evaluate on, -1 for random task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for the environment and the model.",
    )

    args = parser.parse_args()
    main(args)
