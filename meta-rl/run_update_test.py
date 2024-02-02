import matplotlib.pyplot as plt

from read_out_train import plot_progress, read_train_data_individual

for _ in range(10000):
    try:
        (
            returns_train,
            returns_valid,
            returns_mean_train,
            returns_mean_valid,
        ) = read_train_data_individual(my_dir="awake/test", meta_train=False)
        plot_progress(
            returns_train,
            returns_valid,
            returns_mean_train,
            returns_mean_valid,
            title="stats. during standard training",
        )
    except UnboundLocalError:
        print("Waiting for data")
    plt.pause(10)
