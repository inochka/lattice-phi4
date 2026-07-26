import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.lattice import Lattice
from core.utils import get_corr_func_mom_optimized, get_momenta_grid, get_corr_func_mom_parallel

M = 32  # number of lattice sites in every dimension

G_s = [2.0, 5.0, 10.0, 40.0]   # list of g^4 to run simulations for 
gammas = [1.]                  # list of \gamma coefficients of L operator to run simulation for
alpha = 1.                     # \alpha coefficient of L operator
d = 3                          # number of lattice dimensions

DATA_DIRECTORY = Path("./data_enhanced/")   # directory for data saving (in this script - only reference.txt and FILE_PATH below)
FILE_PATH = DATA_DIRECTORY / f"two_point_data_immediate_{d}.csv"
APPEND_EVERY = 10   #  how often we append field configurations making hmc steps (!=1 to avoid autocorrelations in Markov chains)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')



def compute_corr_func(lock, params):
    d, G, gamma, alpha = params
    cfgs = []
    L = Lattice(M, d, alpha, gamma, G)

    logger.info(f"Starting computations for g^4={G}, gamma={gamma}, alpha={alpha}..")

    logger.info(f"Calculating field configurations...")

    #  d = 2 - 1000 iterations, d=3 - 5000 iterations for worm-up.
    #  in result, universal formula - (~d^1.5 to avoid ``dimension curse``) 1000 d^1.5
    n_warmup = int(2000 * np.power(d, 1.5))
    for _ in tqdm(range(n_warmup)):
        # possibly, for higher dimensions it needs more iterations for warm-up
        phi, accepted = L.hmc()

    accepted_num = 0
    # increase number of iterations for higher dimensions
    n_iter = int(8000 * d * np.log2(d)) # 500
    #10000 d ln d

    for i in tqdm(range(n_iter)):
        phi, accepted = L.hmc()
        if accepted:
            accepted_num += 1

        if i % APPEND_EVERY == 0:
            cfgs.append(copy.deepcopy(phi))


    cfgs = np.array(cfgs)
    logger.info(f"Calculating correlation function...")

    momenta_grid = get_momenta_grid(M, d)[:-1]
    corr_f_mom = get_corr_func_mom_parallel(cfgs, momenta_grid)
    results_list = []

    for i in range(M):
        results_list.append({"g^4": G, "gamma": gamma, "D(p)": corr_f_mom.T[0, i],
                             "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})

    df = pd.DataFrame(results_list, columns=["g^4", "gamma", "D(p)", "error", "p"])

    #with lock:
    if os.path.isfile(FILE_PATH):
        df_old = pd.read_csv(FILE_PATH)
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(FILE_PATH, index=False)
    with open(DATA_DIRECTORY / 'reference.txt', 'a+') as f:
        f.write(f"{d},{G},{gamma},{accepted_num / n_iter};\n")

    return f"Computations for g^4={G}, gamma={gamma}, alpha={alpha} finished with acceptance rate {accepted_num / n_iter}!"


tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]


if __name__ == '__main__':
    tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]

    for task in tasks:
        compute_corr_func(None, task)

    #with Manager() as manager:
    #    lock = manager.Lock()
    #    with Pool(processes=1) as pool:
    #        results = list(tqdm(pool.starmap(compute_corr_func, [(None, task) for task in tasks]), total=len(tasks)))

    #    print(results)
