import os.path

from core.lattice import Lattice
from tqdm import tqdm
import numpy as np
import copy
from multiprocessing import Pool
import logging
import pandas as pd
from time import sleep
from pathlib import Path
from core.utils import get_corr_func_mom, get_momenta_grid
from files_to_two_point_num_multiproc import compute_corr_func_from_arr

M = 32

G_s = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
gammas = [1.]
alpha = 1.
d = 3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')

DATA_DIRECTORY = Path("./data_enhanced/")
APPEND_EVERY = 1
FILE_PATH = DATA_DIRECTORY / "two_point_data_immediate.csv"

def compute_corr_func(params):
    d, G, gamma, alpha = params
    cfgs = []
    L = Lattice(M, d, alpha, gamma, G)

    logger.info(f"Starting computations for g^4={G}, gamma={gamma}, alpha={alpha}..")

    logger.info(f"Calculating field configurations...")

    for _ in tqdm(range(1000)):
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

    momenta_grid = get_momenta_grid(M, d)
    corr_f_mom = get_corr_func_mom(cfgs, momenta_grid).T
    results_list = []

    for i in range(M + 1):
        results_list.append({"g^4": G, "gamma": gamma, "D(p)": corr_f_mom.T[0, i],
                             "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})

    df = pd.DataFrame(results_list, columns=["g^4", "gamma", "D(p)", "error", "p"])
    df.to_csv(FILE_PATH)

    with open(DATA_DIRECTORY / 'reference.txt', 'a+') as f:
        f.write(f"{d},{G},{gamma},{accepted_num / n_iter};\n")

    return f"Computations for g^4={G}, gamma={gamma}, alpha={alpha} finished with acceptance rate {accepted_num / n_iter}!"


tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]

for task in tqdm(tasks):
    compute_corr_func(task)
