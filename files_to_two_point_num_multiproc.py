import numpy as np
import os
import re
from core.utils import get_corr_func_mom
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def compute_corr_func_from_file(params):
    print(f"Computing correlation function im momenta picture for {params}...")
    filepath, G, gamma = params
    momenta_grid = 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi
    results_list = []
    print(f"Processing file {filepath}...")

    cfgs = np.load(filepath)[np.arange(0, 10000, 10)]

    corr_f_mom = get_corr_func_mom(cfgs, momenta_grid)

    plt.figure(figsize=(10, 8))
    # plt.scatter(momenta_grid.T[0], corr_f_mom.T[0])
    plt.errorbar(momenta_grid.T[0], corr_f_mom.T[0], yerr=corr_f_mom.T[1], fmt='o')
    plt.show()
    for i in range(M + 1):
        results_list.append({"g^4": G, "gamma": gamma, "D(p)": corr_f_mom.T[0, i],
                             "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})

    return results_list

alpha = 1.
gamma = 1.
G_s = [0.0, 0.5, 1.0, 5.0, 10.0, 20.0, 40.0]

d = 3
M = 32

DATA_DIRECTORY = Path("./data_enhanced/")

results_list = []

if __name__ == '__main__':
    tasks = []
    for entry in os.scandir(DATA_DIRECTORY):
        search_res = re.search(rf"{d}_(.*)_(.*)\.npy", entry.name)
        if not search_res:
            continue
        groups = search_res.groups()

        G = float(groups[0])
        tasks.append((entry.path, G, gamma))
        #if len(tasks) > 1:
        #    break
    print(tasks)
    with Pool(processes=2) as pool:  # Или любое другое количество процессов, которое вы хотите использовать
        res = list(pool.imap(compute_corr_func_from_file, tasks))

    print(res)

    final_results = []

    for result in res:
        final_results += result

    df = pd.DataFrame(final_results, columns=["g^4", "gamma", "D(p)", "error", "p"])

    print(df.head())
    print(len(df))

    df.to_csv(f"data/two_point_{d}.csv")