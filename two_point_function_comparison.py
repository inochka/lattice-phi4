import numpy as np
import os
import re
import pandas as pd
from integrands import two_point_correlator_amputated_w, two_point_correlator_amputated_s, G_xi_s, G_xi_w
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.lines import Line2D

d = 3
M = 16 #8 #32
alpha = 1.
gamma = 1.

momenta_grid = np.array([[p] + [0.] * (d - 1) for p in np.linspace(-np.pi, np.pi, 50)])
two_point_num = pd.read_csv(f'data_enhanced/two_point_data_immediate_{d}.csv')
#two_point_num = pd.read_csv(f'data/two_point_{d}.csv')

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

plt.figure(figsize=(10, 8))

G_s = sorted(two_point_num["g^4"].unique())
used_G_s = []

for G in G_s:

    if G >= 10:
        continue

    #if G < 0.5:
    #    continue

    used_G_s.append(G)

    res = two_point_num.where(two_point_num["g^4"] == G)
    corr_f_mom = res["D(p)"].values
    errors = res["error"].values
    p_s = res["p"].values
    p_s = [p if p < np.pi else p - 2 * np.pi for p in p_s]
    g = np.power(G, 0.25)

    #two_point_analytic_w = two_point_correlator_amputated_w()

    plt.errorbar(p_s, corr_f_mom, yerr=2 * errors, fmt='o', label='_nolegend_', markersize=2.5, color=colors[G])
    gf = []

    for xi in tqdm(momenta_grid):
        gf.append(G_xi_w(alpha=alpha, gamma=gamma, xi=xi)**2 *
                  two_point_correlator_amputated_w(alpha=alpha, gamma=gamma, xi=xi, d=d, g=g))

        #gf.append( G_xi_w(alpha=alpha, gamma=gamma, xi=xi) -
        #           G_xi_s(alpha=alpha, gamma=gamma, xi=xi, g=g)**2 * G_xi_w(alpha=alpha, gamma=gamma, xi=xi)**2 *
        #           two_point_correlator_amputated_s(alpha=alpha, gamma=gamma, xi=xi, d=d, g=g))

    gf = np.array(gf)
    plt.plot(momenta_grid.T[0], gf.T[0], label='_nolegend_', color=colors[G], alpha=0.5)



legend_lines = [Line2D([0], [0], color=colors[G], marker='o', linestyle='-', label=rf'$g^4={G}$') for G in used_G_s]

plt.xlabel(r"$p$", fontsize=20)
plt.ylabel(r"$G_g(p)$", fontsize=20, rotation=0, labelpad=30)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(loc='upper left', shadow=True, fontsize='x-large', handles=legend_lines)
#plt.title("Two-point function comparison (strong coupling)", fontsize=23)
plt.title("Two-point function comparison (weak coupling)", fontsize=23)
plt.grid()
plt.savefig(f"immediate_calc/two_point_comparison_weak_all_{d}.png")
#plt.savefig(f"two_point_comparison_strong_{d}.png")

plt.show()
