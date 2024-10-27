from core.lattice import Lattice
from tqdm import tqdm
import numpy as np
import copy
from multiprocessing import Pool
import logging
from time import sleep
from pathlib import Path

M = 8 #32

#G_s = list(np.power((np.linspace(0, 2, 20)), 4)) + list(np.power((np.linspace(2.1, 8, 20)), 4))
#G_s = np.linspace(0, 50, 200)
#G_s = list(np.linspace(0, 5, 10)) + list(np.linspace(5, 50, 10))

G_s = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
gammas = [1.]
alpha = 1.
d = 3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')

DATA_DIRECTORY = Path("./data_enhanced/")
APPEND_EVERY = 10

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

    with Pool(processes=6) as pool:  # Или любое другое количество процессов, которое вы хотите использовать
        results = list(tqdm(pool.imap(compute_cfgs, tasks), total=len(tasks)))
    # TODO: для экономии места одновременно запускать процесс перереботки файлов в своб энергию / корреляторы?
    #  может, все в одномерных массивах nunpy хранить??

    # TODO: погугли в целом, как можно сжимать numpy массивы. но как будто бы уже реально проще сразупросчитывать.
    #  И типа какой-то процесс поставить на запись в файл. Ну или залочить доступ к файлу у данного процесса.
    print(results)

