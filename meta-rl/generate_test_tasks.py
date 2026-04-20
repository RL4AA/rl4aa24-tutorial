import argparse
import os
import pickle

import numpy as np
import seaborn as sns
import torch
from matplotlib import colors
from matplotlib import pyplot as plt

from maml_rl.envs.helpers import MamlHelpers


def main(args):
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Define file location and name
    output_folder = args.output_folder
    filename = args.filename

    # Create an instance of MamlHelpers and sample tasks
    num_tasks = args.num_tasks
    tasks = []
    max_variation = args.max_variation

    while len(tasks) < num_tasks:
        sampled_task = MamlHelpers().sample_tasks(
            num_tasks=1, num_random_quads=args.num_random_quads
        )[
            0
        ]  # Assuming sample_tasks returns a list of tasks
        task_goal_values = sampled_task["goal"][0]

        max_value = np.max(task_goal_values)
        min_value = np.min(task_goal_values)

        if max_value < max_variation and abs(min_value) < max_variation:
            tasks.append(sampled_task)

    tasks_new = [MamlHelpers().get_origin_task(idx=0)] + [
        {"id": task["id"] + 1, **task} for task in tasks
    ]
    # print(tasks_new)

    num_tasks = len(tasks_new)

    # Determine the number of rows and columns for the subplot grid
    cols = int(np.ceil(np.sqrt(num_tasks)))
    rows = int(np.ceil(num_tasks / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Find the global min and max across all tasks for normalization
    all_values = [task["goal"][0] for task in tasks_new]
    global_min = min(map(np.min, all_values))
    global_max = max(map(np.max, all_values))

    # Get a triangular mask for the lower triangle of the matrix
    resp_mat = tasks_new[0]["goal"][0]
    mask = 1 - np.tri(resp_mat.shape[0])

    # Define a common normalization
    # norm = colors.Normalize(vmin=global_min, vmax=global_max)
    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])

    for nr, task in enumerate(tasks_new):
        response_matrix = task["goal"][0]
        _ = sns.heatmap(
            response_matrix,
            mask=mask,
            cmap="viridis",
            vmax=global_max,
            vmin=global_min,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.1},
            ax=axs[nr],
            cbar=not nr,
            cbar_ax=cbar_ax,
        )
        axs[nr].set_title(f"Task {nr}")
        axs[nr].set_xlabel("Corrector index")
        axs[nr].set_ylabel("BPM index")

    # Turn off axes for any empty subplots
    for nr in range(num_tasks, len(axs)):
        axs[nr].axis("off")

    axs[0].set_title("Origin task")

    # Add a global colorbar
    fig.subplots_adjust(top=0.9, left=0.1, right=0.85)
    # cbar_ax = fig.add_axes([0.9, 0.05, 0.15, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout(rect=[0, 0, 0.8, 1])
    fig.suptitle("Response matrices of the tasks")
    fname_stripped = filename.split(".")[0]
    plt.savefig(f"img/task_overview_{fname_stripped}.pdf", bbox_inches="tight")
    plt.savefig(f"img/task_overview_{fname_stripped}.png", dpi=300)
    if args.show_plot:
        plt.show()

    # Construct the full file path
    full_path = os.path.join(output_folder, filename)

    # Save the tasks using pickle
    with open(full_path, "wb") as fp:
        pickle.dump(tasks_new, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, default="configs/")
    parser.add_argument("--filename", type=str, default="evaluation_tasks.pkl")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument(
        "--num-random-quads",
        type=int,
        default=10,
        help="number of quadrupole strengths to vary",
    )
    parser.add_argument(
        "--max-variation",
        type=float,
        default=20.0,
        help="maximal value of the response matrix",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    main(args)
