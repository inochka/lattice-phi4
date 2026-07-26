from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from analytical_expressions import two_point_correlator_amputated_w, two_point_correlator_amputated_s, G_xi_s, G_xi_w

d = 3
M = 8
alpha = 1.
gamma = 1.

momenta_grid = np.array([[p] + [0.] * (d - 1) for p in np.linspace(-np.pi, np.pi, 50)])
two_point_num = pd.read_csv(f'data_enhanced/two_point_data_immediate_{d}_averaged.csv')

DATA_DIRECTORY = Path("./data_enhanced/")

colors = {
    0.0: "blue",
    0.5: "green",
    1.0: "red",
    2.0: "purple",
    5.0: "gray",
    10.0: "cyan",
    20.0: "brown",
    40.0: "crimson"
}

PLOT_CONDITIONS = {
    "strong": lambda G: G >= 0.5,
    "weak": lambda G: G < 10
}

PLOT_REGIME = "strong"  # place weak or strong mode comparison here

theory_values_filename = f"theory_2_point_{d}_{PLOT_REGIME}.npy"

plt.figure(figsize=(10, 8))

G_s = sorted(two_point_num["g^4"].unique())
used_G_s = []
gfs = []

print(f"Starting computation in {PLOT_REGIME} regime for couplings: {[G for G in G_s if PLOT_CONDITIONS[PLOT_REGIME](G)]}")

# рассчитываем и строим теоретические значения
if os.path.exists(theory_values_filename):
    gf = np.load(theory_values_filename)
else:
    for G in G_s:
        if not PLOT_CONDITIONS[PLOT_REGIME](G):
            continue
        gf = []
        g = np.power(G, 0.25)

        for xi in tqdm(momenta_grid):
            if PLOT_REGIME == "weak":
                gf.append(G_xi_w(alpha=alpha, gamma=gamma, xi=xi)**2 *
                          two_point_correlator_amputated_w(alpha=alpha, gamma=gamma, xi=xi, d=d, g=g))
            elif PLOT_REGIME == "strong":
                gf.append( G_xi_w(alpha=alpha, gamma=gamma, xi=xi) -
                           G_xi_s(alpha=alpha, gamma=gamma, xi=xi, g=g)**2 * G_xi_w(alpha=alpha, gamma=gamma, xi=xi)**2 *
                           two_point_correlator_amputated_s(alpha=alpha, gamma=gamma, xi=xi, d=d, g=g))
            else:
                print(f"Incorrect plot comparison regime: {PLOT_REGIME}. Chose one of 'strong' or 'weak'")

        gf = np.array(gf)
        gfs.append(gf)
        plt.plot(momenta_grid.T[0], gf.T[0], label='_nolegend_', color=colors[G], alpha=0.5,)

    gfs = np.array(gfs)
    np.save(f"theory_2_point_{d}_{PLOT_REGIME}", gfs)

# строим экспериментальные значения
for G in G_s:
    if not PLOT_CONDITIONS[PLOT_REGIME](G):
        continue

    used_G_s.append(G)

    res = two_point_num.where(two_point_num["g^4"] == G)
    corr_f_mom = res["D(p)"].values
    errors = res["error"].values
    p_s = res["p"].values
    p_s = [p if p < np.pi else p - 2 * np.pi for p in p_s]

    plt.errorbar(p_s, corr_f_mom, yerr=errors, fmt='o', label='_nolegend_', markersize=2.5, color=colors[G])

legend_lines = [Line2D([0], [0], color=colors[G], marker='o', linestyle='-', label=rf'$g^4={G}$') for G in used_G_s]

plt.xlabel(r"$p$", fontsize=20)
plt.ylabel(r"$G_g(p)$", fontsize=20, rotation=0, labelpad=30)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(loc='upper left', shadow=True, fontsize='x-large', handles=legend_lines)
plt.title(f"Two-point function comparison ({PLOT_REGIME} coupling)", fontsize=23)
plt.grid()
plt.savefig(f"immediate_calc/two_point_comparison_{PLOT_REGIME}_{d}.png")

plt.show()
