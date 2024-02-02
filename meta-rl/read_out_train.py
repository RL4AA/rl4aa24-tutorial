import argparse
import glob
import os
import pickle
from collections import defaultdict
from math import exp
from turtle import color
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from maml_rl.utils.reinforcement_learning import get_returns
from sympy import root

# from maml_rl.utils.torch_utils import to_numpy


def read_train_data(my_dir="awake/test_me/train/progress"):
    """
    Reads training data from a specified directory and computes mean returns.

    Parameters:
    my_dir (str): Directory from which to read the training data files.

    Returns:
    tuple: Tuple containing lists of returns
        and rolling mean returns for both training and validation.
    """
    files = glob.glob(os.path.join(my_dir, "training*"))
    files.sort()

    returns_train, returns_valid = [], []
    returns_mean_train, returns_mean_valid = [], []
    nr_total_interactions = 0
    for full_path in files:
        try:
            with open(full_path, "rb") as file:
                data_train, data_valid = pickle.load(file)

            if data_train:
                returns_train.append(np.mean(get_returns(data_train)))
                train_episodes_len = np.reshape(get_episode_lengths(data_train), -1)
                nr_total_interactions += np.sum(train_episodes_len, axis=0)
                # print('training interactions:', nr_total_interactions)
            if data_valid:
                returns_valid.append(np.mean(get_returns(data_valid)))

            returns_mean_train.append(np.mean(returns_train[-20:]))
            returns_mean_valid.append(np.mean(returns_valid[-20:]))
        except IOError as e:
            print(f"Error opening file {full_path}: {e}")
        except pickle.UnpicklingError as e:
            print(f"Error unpickling file {full_path}: {e}")

    return (
        returns_train,
        returns_valid,
        returns_mean_train,
        returns_mean_valid,
        nr_total_interactions,
    )


def accumulate_data(data_list, data_individual):
    for task, data in enumerate(data_list):
        data_individual[task].append(data)


def read_train_data_individual(my_dir="awake/test_me/train/progress"):
    """
    Reads training data from a specified directory and
    accumulates task-specific data for both training and validation.

    Parameters:
    my_dir (str): Directory from which to read the training data files.

    Returns:
    tuple: Tuple containing dictionaries with
        task-specific data for training and validation.
    """
    files = glob.glob(os.path.join(my_dir, "training*"))
    files.sort()

    data_train_individual = defaultdict(list)
    data_valid_individual = defaultdict(list)

    for full_path in files:
        try:
            with open(full_path, "rb") as file:
                data_train, data_valid = pickle.load(file)

            # Accumulate task-specific data
            if data_train:
                accumulate_data(data_train, data_train_individual)
            if data_valid:
                accumulate_data(data_valid, data_valid_individual)

        except IOError as e:
            print(f"Error opening file {full_path}: {e}")
        except pickle.UnpicklingError as e:
            print(f"Error unpickling file {full_path}: {e}")

    return data_train_individual, data_valid_individual


def plot_progress(
    returns_train: List[float],
    returns_valid: List[float],
    returns_mean_train: List[float],
    returns_mean_valid: List[float],
    title: str = "Statistics During Training from specific Task",
    ax: Optional[plt.Axes] = None,
    save_folder: Optional[str] = None,
    file_name: str = "plot",
    show_plot: bool = False,
    label_prefix: str = "",
    file_formats: List[str] = ["pdf", "png"],
):
    """
    Plots the progress of training and validation returns,
    including their mean values.

    Parameters:
    - returns_train, returns_valid, returns_mean_train,
        returns_mean_valid (list of float): Data points.
    - title (str): Title of the plot.
    - ax (matplotlib Axes, optional):
        Axis on which to plot. If None, uses the current axis.
    - save_folder (str, optional):
        Folder to save the plot. If None, the plot is not saved.
    - file_name (str): Base name for saved files.
    - show_plot (bool): If True, displays the plot.
    - label_prefix (str): Prefix for the plot labels.
    - file_formats (list of str): Formats in which to save the plot.
    """
    if not any([returns_train, returns_valid, returns_mean_train, returns_mean_valid]):
        print("No data to plot.")
        return

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    # Plotting data with optional label prefix
    if returns_train:
        train_plot = ax.plot(
            returns_train,
            # label=label_prefix + "Train",
            linestyle="-",
            alpha=0.3,
        )
    if returns_valid:
        valid_plot = ax.plot(
            returns_valid,
            # label=label_prefix + "Valid",
            linestyle="--",
            alpha=0.3,
        )
    if returns_mean_train:
        ax.plot(
            returns_mean_train,
            label=label_prefix + "Train",
            color=train_plot[0].get_color(),
            lw=2,
        )
    if returns_mean_valid:
        ax.plot(
            returns_mean_valid,
            label=label_prefix + "Valid",
            color=valid_plot[0].get_color(),
            lw=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Batches")
    ax.set_ylabel("Returns")
    ax.legend(loc="lower right")
    ax.grid(True)

    if save_folder:
        for fmt in file_formats:
            file_path = f"{save_folder}/{file_name}.{fmt}"
            try:
                plt.gcf().savefig(file_path, format=fmt)
            except ValueError as e:
                print(f"Unsupported format '{fmt}': {e}")
            except Exception as e:
                print(f"Error saving file '{file_path}': {e}")

    if show_plot:
        plt.draw()
        plt.pause(0.1)

    return ax


def get_episode_lengths(episodes):
    return [episode.lengths for episode in episodes]


def get_episode_lengths_mean(episodes):
    return [np.mean(episode.lengths) for episode in episodes]


def plot_progress_individual(
    data_train_individual,
    data_valid_individual,
    title="stats. during meta training",
    save_folder=None,
):
    fig, axs = plt.subplots(2, sharex=True)

    for task in data_train_individual:
        axs[0].plot(
            (get_episode_lengths_mean(data_valid_individual[task])),
            label=f"task: {task}",
        )
        axs[1].plot((get_returns(data_train_individual[task])), label=f"task: {task}")
    axs[0].legend()
    plt.show()


def setup_and_plot(base_folder, experiment_name, experiment_type, ax, label_prefix):
    experiment_folder = os.path.join(base_folder, experiment_name)
    save_progress_data_dir = os.path.join(
        experiment_folder, experiment_type, "progress"
    )

    # Read training data
    (
        returns_train,
        returns_valid,
        returns_mean_train,
        returns_mean_valid,
        _,
    ) = read_train_data(save_progress_data_dir)

    if any([returns_train, returns_valid, returns_mean_train, returns_mean_valid]):
        # Plotting the progress
        plot_progress(
            returns_train,
            returns_valid,
            returns_mean_train,
            returns_mean_valid,
            title="Comparing different training approaches",
            ax=ax,
            label_prefix=label_prefix,
        )
    else:
        print(f"No data available for plotting in experiment '{experiment_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read out training data.")

    parser.add_argument(
        "--root-folder",
        type=str,
        default="awake/",
        help="Path to the root folder",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="test_me",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="train",
        help="Experiment type: train or test.",
    )

    args = parser.parse_args()

    progress_folder = os.path.join(
        args.root_folder, args.experiment_name, args.experiment_type, "progress"
    )

    (
        returns_train,
        returns_valid,
        returns_mean_train,
        returns_mean_valid,
        nr_total_interactions,
    ) = read_train_data(my_dir=progress_folder)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plot_progress(
        returns_train,
        returns_valid,
        returns_mean_train,
        returns_mean_valid,
        title=f"Statistics for exp: {args.experiment_type}, "
        + f"total {nr_total_interactions} steps",
        ax=ax,
    )
    ax.set_ylim(-120, 0)  # For tutorial purposes

    data_train_individual, data_valid_individual = read_train_data_individual(
        my_dir=progress_folder
    )
    plot_progress_individual(data_train_individual, data_valid_individual)
