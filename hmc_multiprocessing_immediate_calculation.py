import copy
import logging
from pathlib import Path
from multiprocessing import Pool, Lock, Manager
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from core.lattice import Lattice
from core.utils import get_corr_func_mom, get_corr_func_mom_optimized, get_momenta_grid

M = 32

G_s = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
#G_s = [0.0]
gammas = [1.]
alpha = 1.
d = 3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')

DATA_DIRECTORY = Path("./data_enhanced/")
APPEND_EVERY = 1
FILE_PATH = DATA_DIRECTORY / f"two_point_data_immediate_{d}.csv"

def compute_corr_func(lock, params):
    d, G, gamma, alpha = params
    cfgs = []
    L = Lattice(M, d, alpha, gamma, G)

    logger.info(f"Starting computations for g^4={G}, gamma={gamma}, alpha={alpha}..")

    logger.info(f"Calculating field configurations...")

    for _ in tqdm(range(5000)):
        # возможно, для более высоких размерностей ему нужно дольше прогреваться.
        # !! да, так и есть. для тройки - 5000, для двойки хватало и 1000
        phi, accepted = L.hmc()

    accepted_num = 0
    n_iter = 10000

    for i in tqdm(range(n_iter)):
        phi, accepted = L.hmc()
        if accepted:
            accepted_num += 1

        if i % APPEND_EVERY == 0:
            cfgs.append(copy.deepcopy(phi))

    cfgs = np.array(cfgs)

    logger.info(f"Calculating correllation function...")

    momenta_grid = get_momenta_grid(M, d)[:-1]
    corr_f_mom = get_corr_func_mom_optimized(cfgs, momenta_grid)
    results_list = []

    for i in range(M):
        results_list.append({"g^4": G, "gamma": gamma, "D(p)": corr_f_mom.T[0, i],
                             "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})

    df = pd.DataFrame(results_list, columns=["g^4", "gamma", "D(p)", "error", "p"])

    with lock:
        if os.path.isfile(FILE_PATH):
            df_old = pd.read_csv(FILE_PATH)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(FILE_PATH)
        with open(DATA_DIRECTORY / 'reference.txt', 'a+') as f:
            f.write(f"{d},{G},{gamma},{accepted_num / n_iter};\n")

    return f"Computations for g^4={G}, gamma={gamma}, alpha={alpha} finished with acceptance rate {accepted_num / n_iter}!"


tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]

#for task in tqdm(tasks):
#    compute_corr_func(task)

if __name__ == '__main__':
    tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]

    with Manager() as manager:
        lock = manager.Lock()
        with Pool(processes=1) as pool:
            results = list(tqdm(pool.starmap(compute_corr_func, [(lock, task) for task in tasks]), total=len(tasks)))

        print(results)
