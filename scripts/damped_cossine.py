import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# from matplotlib import cm

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[sc]{mathpazo}",
    }
)


def distance(
    x: NDArray | float,
    y: NDArray | float,
    xc: NDArray | float,
    yc: NDArray | float,
) -> NDArray | float:
    """
    Computes the distance between (x, y) and (xc, yc)
    """

    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def gaussian(
    amp: NDArray | float,
    r: NDArray | float,
    sigma: NDArray | float,
) -> NDArray | float:
    """
    A Gaussian in polar coordinates with amplitude amp and width sigma
    """

    return amp * np.exp(-((r / sigma) ** 2))


def damped_cossine(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    r = distance(x, y, 0, 0)
    return np.cos(9 * np.pi * r) * gaussian(1.0, r, 0.4)


# Create the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Setting up labels
ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$f(x, y)$")
x = y = np.linspace(-1, 1, 500)
x, y = np.meshgrid(x, y)

# Ploting the fig
ax.plot_trisurf(
    x.flatten(),
    y.flatten(),
    damped_cossine(x, y).flatten(),
    # cmap=cm.jet,
)

# Saving the fig
script_dir = os.path.dirname(__file__)
workspace_dir = os.path.split(script_dir)[0]
out_file = os.path.join(workspace_dir, "imagens/damped_cossine.png")
fig.savefig(out_file, dpi=300)
plt.show()
