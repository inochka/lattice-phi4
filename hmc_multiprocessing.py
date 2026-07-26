from core.lattice import Lattice
from tqdm import tqdm
import numpy as np
import copy
from multiprocessing import Pool
import logging
from pathlib import Path

M = 8 #32

G_s = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]  # list of g^4 to run simulations for 
gammas = [1.] # list of \gamma coefficients of L operator to run simulation for
alpha = 1.    # \alpha coefficient of L operator
d = 3         # number of lattice dimensions

DATA_DIRECTORY = Path("./data_enhanced/")   # directory for data saving (.npy arrays and reference.txt)
APPEND_EVERY = 10  # how often we append field configurations making hmc steps (!=1 to avoid autocorrelations in Markov chains)
PROCESSES_NUM = 6  # num of processes to perform parallel computations 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')



def compute_cfgs(params):
    d, G, gamma, alpha = params
    cfgs = []
    L = Lattice(M, d, alpha, gamma, G)

    logger.info(f"Starting computations for g^4={G}, gamma={gamma}, alpha={alpha}..")

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

    np.save(DATA_DIRECTORY / f'{d}_{G}_{gamma}', cfgs)

    with open(DATA_DIRECTORY / 'reference.txt', 'a+') as f:
        f.write(f"{d},{G},{gamma},{accepted_num / n_iter};\n")

    return f"Computations for g^4={G}, gamma={gamma}, alpha={alpha} finished with acceptance rate {accepted_num / n_iter}!"


if __name__ == '__main__':
    tasks = [(d, G, gamma, alpha) for G in G_s for gamma in gammas]

    with Pool(processes=PROCESSES_NUM) as pool: 
        results = list(tqdm(pool.imap(compute_cfgs, tasks), total=len(tasks)))

    print(results)

