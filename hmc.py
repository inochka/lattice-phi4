from core.lattice import Lattice
from tqdm import tqdm
import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


N = 32

G_s = np.linspace(0, 1.5, 20)
#gammas = [0.1, 0.5, 1, 2]
gammas = [1]
alpha = 1.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')


def compute_cfgs(d, G, gamma, alpha):
    cfgs = []
    L = Lattice(N, d, alpha, gamma, G)

    print(f"Starting computations for g^4={G}, gamma={gamma}, alpha={alpha}..")

    for _ in range(1000):
        phi, accepted = L.hmc()

    accepted_num = 0
    n_iter = 10000

    for i in range(n_iter):
        phi, accepted = L.hmc()
        if accepted:
            accepted_num += 1

        if i % 10 == 0:
            cfgs.append(copy.deepcopy(phi))

    cfgs = np.array(cfgs)

    np.save(f'data/{d}_{G}_{gamma}', cfgs)

    with open('data/reference.txt', 'a+') as f:
        f.write(f"{d},{G},{gamma},{accepted_num / n_iter};\n")

    return f"Computations for g^4={G}, gamma={gamma}, alpha={alpha} finished with acceptance rate {accepted_num / n_iter}!"


futures = []
results = []

with ThreadPoolExecutor(max_workers=2) as executor:
    for d in range(2, 3):
        for G in G_s:
            for gamma in gammas:
                futures.append(executor.submit(compute_cfgs, d, G, gamma, alpha))

    # Инициализация tqdm в главном потоке
progress = tqdm(total=len(futures), desc="Processing", unit="task")

# Ожидание и обновление прогресса по мере завершения задач
for future in as_completed(futures):
    result = future.result()  # Получение результата задачи
    results.append(result)
    progress.update(1)  # Обновление прогресс-бара на 1

progress.close()  # Закрытие прогресс-бара после завершения всех задач

#results = [future.result() for future in concurrent.futures.as_completed(futures)]
print(results)

