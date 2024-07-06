import numpy as np
import os
import re
from core.utils import get_corr_func_mom
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

alpha = 1.
gamma = 1.
G_s = [0.0, 0.5, 1.0, 5.0, 10.0, 20.0, 40.0]

d = 3
M = 32

#momenta_grid = np.array([[p, 0.] for p in np.linspace(0, 2 * np.pi, 200)])

momenta_grid = 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi #- np.pi

print(momenta_grid)

DATA_DIRECTORY = Path("./data_enhanced/")

results_list = []

for entry in os.scandir(DATA_DIRECTORY):
    search_res = re.search(rf"{d}_(.*)_(.*)\.npy", entry.name)
    if search_res:
        print(f"Processing file {entry.name}...")
        groups = search_res.groups()

        G = float(groups[0])
        if G not in G_s:
            continue

        #if G > 5:
        #    continue

        cfgs = np.load(entry.path)[np.arange(0, 10000, 10)]
        print(cfgs.shape)

        corr_f_mom = get_corr_func_mom(cfgs, momenta_grid)

        plt.figure(figsize=(10, 8))
        #plt.scatter(momenta_grid.T[0], corr_f_mom.T[0])
        plt.errorbar(momenta_grid.T[0], corr_f_mom.T[0], yerr=corr_f_mom.T[1], fmt='o')
        plt.show()
        for i in range(M + 1):
            results_list.append({"g^4": float(groups[0]), "gamma": float(groups[1]), "D(p)": corr_f_mom.T[0, i],
                                 "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})


results = pd.DataFrame(results_list, columns=["g^4", "gamma", "D(p)", "error", "p"])

print(results.head())
print(len(results))

results.to_csv(f"data/two_point_{d}.csv")