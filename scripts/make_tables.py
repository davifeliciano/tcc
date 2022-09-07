from fileinput import filename
from pathlib import Path
import pandas as pd


project_dir = Path(__file__).parent.parent
results_dir = project_dir.joinpath("results")
lattice_names = {"crs2": r"\ch{CrS2}", "crse2": r"\ch{CrSe2}"}

for lattice_name, chem_formula in lattice_names.items():
    methods = ("Algorítmo Genético", r"\textit{Dual Annealing}")
    files = sorted(list(results_dir.glob(f"{lattice_name}*.csv")), reverse=True)
    dfs = {
        method: pd.read_csv(file, index_col=0, usecols=(0, 1, 2))
        for method, file in zip(methods, files)
    }

    # For each df in dfs, change columns and indexes
    for df in dfs.values():
        df.columns = ("1ª Ordem", "3ª Ordem")
        df.index = (
            r"$f$",
            r"$a$",
            r"$E_F$",
            r"$\Delta$",
            r"$\lambda_c$",
            r"$\lambda_v$",
            r"$\gamma_0$",
            r"$\gamma_1$",
            r"$\gamma_2$",
            r"$\gamma_3$",
            r"$\gamma_4$",
            r"$\gamma_5$",
            r"$\gamma_6$",
        )

    df = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    df = df.drop(index=r"$a$")

    basename = f"{lattice_name}_tabular"
    filename = project_dir.joinpath(f"{basename}.tex")
    df.to_latex(filename, na_rep="", escape=False, decimal=",", multicolumn_format="c")
