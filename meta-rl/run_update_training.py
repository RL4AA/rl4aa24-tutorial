import argparse
import os
import pickle

import matplotlib.pyplot as plt

from policy_test import _layout_verficication_plot, verify
from read_out_train import plot_progress, read_train_data

# Define the experiment
parser = argparse.ArgumentParser(description="Read out training data.")

parser.add_argument(
    "--base-folder",
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
    help="Experiment type, typically train or test.",
)
parser.add_argument(
    "--evaluation-tasks",
    type=str,
    default="configs/evaluation_tasks.pkl",
)

args = parser.parse_args()

base_folder = args.base_folder
experiment_name = args.experiment_name
experiment_type = args.experiment_type

experiment_folder = os.path.join(base_folder, experiment_name)
save_progress_data_dir = os.path.join(experiment_folder, experiment_type, "progress")

# Read the tasks from a pre-defined file
evaluation_tasks_file = args.evaluation_tasks
with open(evaluation_tasks_file, "rb") as input_file:
    tasks = pickle.load(input_file)


tasks_selected = tasks
last_processed = None

if experiment_type == "train":
    fig, axes = _layout_verficication_plot(n_tasks=len(tasks_selected))
fig_progress, ax_progress = plt.subplots(1, 1)
for _ in range(10000):
    try:
        # Read data
        new_data = read_train_data(save_progress_data_dir)

        # Check if the data is new or updated
        if new_data is not last_processed:
            last_processed = new_data
            # interactions, episodes = get_number_of_interactions()

            (
                returns_train,
                returns_valid,
                returns_mean_train,
                returns_mean_valid,
                nr_total_interactions,
            ) = new_data

            # Plot the data
            ax_progress.clear()
            plot_progress(
                returns_train,
                returns_valid,
                returns_mean_train,
                returns_mean_valid,
                title="stats. during meta training,"
                + f" interactions: {nr_total_interactions}",
                ax=ax_progress,
                show_plot=False,
            )
            if experiment_type == "train":
                # Reuse the canvas
                verify(
                    tasks=tasks_selected,
                    episodes=20,
                    fig=fig,
                    ax=axes,
                    show_success_rate=True,
                    show_plot=False,
                )
            # Redraw the figure
            fig_progress.canvas.draw()
            fig_progress.canvas.flush_events()
            if experiment_type == "train":
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(0.1)
        else:
            print("No new data available")

    except FileNotFoundError:
        print("Data file not found, waiting for data")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Pause for 10 seconds
    plt.pause(10)
