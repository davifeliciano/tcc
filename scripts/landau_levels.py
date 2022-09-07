from fileinput import filename
from pathlib import Path
from turtle import title
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from mpl_config import latex_preamble

# Setting up Matploltib
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
    }
)

REDUCED_PLANCK = 6.5821e-16
ELEM_CHARGE = 1.6021e-19
CRS2_LATTICE = 3.022302679
CRSE2_LATTICE = 3.167287237


def landau_levels(
    lattice_const: float,
    delta: float,
    lambda_so: float,
    gamma_0: float,
    valley_index: int,
    mag_field: NDArray,
    spin: str = "down",
    n: int = 1,
) -> Tuple[NDArray, NDArray]:
    """
    Returns the sorted eigenvalues of for the first order k.p hamiltonian as a
    function of the fitted parameters, the spin of the considered state and the
    normal magnetic field applied over the lattice

    Parameters
    ----------
    lattice_const : float
        The lattice parameter of the crystal
    delta : float
        Fitted parameter: the bandgap of the crystal on the K (or K') valley
    lambda_so : float
        Fitted parameter: spin-orbit coupling parameter
    gamma_0 : float
        Fitted parameter.
    valley_index : int
        1 for results on the K valley, -1 for results on the K' valley
    mag_field : NDArray
        Normal magnetic field applied over the lattice, in Teslas
    spin : str, optional
        The spin of the considered state, "up" or "down", by default "down"
    n : int, optional
        The state to evaluate. Range is [0, Inf). By default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        the sorted eigenvalues of for the first order k.p hamiltonian

    Raises
    ------
    ValueError
        If the valley_index is not in (-1, 1)
        If the spin is not in ("up", "down")
    """
    if valley_index not in (1, -1):
        raise ValueError("invalid valley index. Must be 1 or -1")
    if spin not in ("up", "down"):
        raise ValueError("Invalid value for spin. Must be 'up' or 'down'")
    if spin == "up":
        spin = 1
    if spin == "down":
        spin = -1

    n = abs(int(n))
    mag_lenght = np.sqrt(REDUCED_PLANCK / (ELEM_CHARGE * mag_field))
    lambda_valley_spin = lambda_so * valley_index * spin
    omega = np.sqrt(2) / mag_lenght
    sqrt_first_term = 0.25 * (delta - lambda_valley_spin) ** 2
    sqrt_second_term = n * (gamma_0 * lattice_const * omega) ** 2
    square_root = np.sqrt(sqrt_first_term + sqrt_second_term)
    return (
        0.5 * lambda_valley_spin - square_root,
        0.5 * lambda_valley_spin + square_root,
    )


project_dir = Path(__file__).parent.parent
images_dir = project_dir.joinpath("imagens")
results_dir = project_dir.joinpath("results")
lattices = {"crs2": CRS2_LATTICE, "crse2": CRSE2_LATTICE}
titles = {"crs2": r"\ch{CrS2}", "crse2": r"\ch{CrSe2}"}
mag_field = np.linspace(0, 10, 100)


for lattice_name, lattice_const in lattices.items():

    # Reading files
    file = results_dir.joinpath(f"{lattice_name}_genetic_algorithm_order_13.csv")
    df = pd.read_csv(file, index_col=0)
    delta, lambda_c, lambda_v, gamma_0 = df.loc["delta":"gamma_0", "order_1"]

    # Creating figures
    fig, axes = plt.subplots(1, 2)
    ax_titles = ("Valência", "Condução")
    lambdas = (lambda_v, lambda_c)
    lambda_labels = (r"\Psi_V", r"\Psi_C")

    for i, (ax, ax_title, lambda_label) in enumerate(
        zip(axes, ax_titles, lambda_labels)
    ):
        ax.set(
            title=ax_title,
            ylabel=r"Energia (\si{\eV})" if i == 0 else None,
            xlabel=r"$B$ (\si{\tesla})",
        )

        spins = ("down", "up")
        spin_labels = (r"\downarrow", r"\uparrow")
        colors = ("blue", "red")

        for spin, color, spin_label in zip(spins, colors, spin_labels):
            for n in range(10):
                eigen = landau_levels(
                    lattice_const=lattice_const,
                    delta=delta,
                    lambda_so=lambda_v,
                    gamma_0=gamma_0,
                    valley_index=1,
                    mag_field=mag_field,
                    spin=spin,
                    n=n,
                )[i]

                ax.plot(
                    mag_field,
                    eigen,
                    color=color,
                    linewidth=1.0,
                    label=r"$\expval{\hat{H}}{" + f"{lambda_label},{spin_label}" + "}$",
                )

        # Removing repeated entries from legend
        handles, eigen_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(eigen_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), framealpha=1.0)

    basename = f"{lattice_name}_landau_levels"
    filename = images_dir.joinpath(f"{basename}.png")
    plt.savefig(filename, dpi=300)
