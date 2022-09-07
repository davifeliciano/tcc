from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_config import latex_preamble, xtick_label_formatter

# Setting up Matploltib
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
    }
)

Y_MARGIN = 0.1


def get_k_k_index(ks: NDArray) -> int:
    (k_k_index,) = np.where(ks[:, 0] == 0.0)
    return k_k_index[0]


def get_plot_domain(ks: NDArray) -> NDArray:
    k_k_index = get_k_k_index(ks)
    k_gamma = ks[0]
    k_k = ks[k_k_index]
    k_m = ks[-1]
    norms = np.apply_along_axis(norm, 1, ks - k_k)
    first_region_norms = norms[:k_k_index]
    second_region_norms = norms[k_k_index:]
    first_region_xs = -first_region_norms / norm(k_gamma - k_k)
    second_region_xs = second_region_norms / norm(k_m - k_k)
    return np.concatenate((first_region_xs, second_region_xs))


project_dir = Path(__file__).parent.parent
data_dir = project_dir.joinpath("data")
images_dir = project_dir.joinpath("imagens")
crs2_file = data_dir.joinpath("mos2_data.csv")

# Reading data from file
df = pd.read_csv(crs2_file)
ks = df.loc[:, "kx":"kz"].to_numpy()
energies = df.loc[:, "e1":"e4"].to_numpy()
sorted_energies = np.sort(energies)

# Creating figures
fig, ax = plt.subplots()

# Setting up the axes
ylim = (
    np.min(sorted_energies[:, 0]) - Y_MARGIN,
    np.max(sorted_energies[:, -1]) + Y_MARGIN,
)

ax.set(
    title=r"\ch{MoS2}",
    ylabel=r"Energia (\si{\eV})",
    ylim=ylim,
)

ax.xaxis.set_major_formatter(plt.FuncFormatter(xtick_label_formatter))
plot_domain = get_plot_domain(ks)
fmts = ("b-", "r-", "b--", "r--")
labels = (  # Ascending order of energy
    r"$\expval{\hat{H}}{\Psi_V,\downarrow}$",
    r"$\expval{\hat{H}}{\Psi_V,\uparrow}$",
    r"$\expval{\hat{H}}{\Psi_C,\downarrow}$",
    r"$\expval{\hat{H}}{\Psi_C,\uparrow}$",
)

for i, (fmt, label) in enumerate(zip(fmts, labels)):
    ax.plot(plot_domain, sorted_energies[:, i], fmt, label=label)

ax.legend()

# Saving figure
basename = "mos2_bands"
filename = images_dir.joinpath(f"{basename}.png")
plt.savefig(filename, dpi=300)
