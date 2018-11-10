from cmfpy import CMF
from cmfpy import datasets
import matplotlib.pyplot as plt
import numpy as np
import palettable
import itertools


# List of all dataset objects.
all_datasets = [
    datasets.Synthetic(n_components=5),
    datasets.SongbirdHVC(),
]

all_algorithms = [
    # "mult",
    "bcd",
    # "gd",
]

# Tunable parameters shared by all datasets/algorithms.
model_options = {
    "n_components": 5,
    "maxlag": 10,
    "tol": 0,
    # "max_time": 20.0,  # TODO -- make this an option.
}
plot_options = {
    "lw": 2,
    "alpha": 0.8,
}

# Colors for each algorithm.
colors = palettable.tableau.BlueRed_6.mpl_colors

# Create one subplot for each dataset.
fig, axes = plt.subplots(1, len(all_datasets), sharey=True)
axes = np.atleast_1d(axes)

# Iterate over datasets.
for data, ax in zip(all_datasets, axes):

    # Apply all algorithms to each dataset.
    for method, color in zip(all_algorithms, colors):

        # Create model.
        model = CMF(alg_name=method, **model_options)
        model.fit(data.generate())

        # Plot learning curve.
        # ax.plot(model.time_hist, model.loss_hist,
        #         color=color, label=method, **plot_options)
        ax.plot(model.loss_hist,
                color=color, label=method, **plot_options)
        ax.set_title(data.name)

# Format subplots
axes[0].set_ylabel("loss")
for ax in axes:
    ax.set_xlabel("iterations")
axes[-1].legend()
fig.tight_layout()
# fig.savefig("01_alg_comparison.pdf")

plt.show()
