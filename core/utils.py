import logging
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from itertools import product
from multiprocessing import Pool, cpu_count, shared_memory, Lock, Manager

import numpy as np
# import numba
from joblib import Parallel, delayed, Memory
# from numba import njit
from tqdm import tqdm

logger = logging.getLogger(__name__)

# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.INFO)


def cross_validation_mean_error_np(samples: np.ndarray, k: int = 100):
    """Return mean and estimated lower error bound using k-fold cross-validation."""
    np.random.shuffle(samples)
    folds = np.array_split(samples, k)  # Делим на фолды, учитывая размер выборки
    means = []

    for i in tqdm(range(k)):
        #validation_samples = folds[i]
        train_samples = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        means.append(train_samples.mean(axis=0))
        #means.append(validation_samples.mean(axis=0))


    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt(k) * np.std(means, axis=0, ddof=1)

    return np.array([mean, error])

def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in tqdm(range(samples.shape[0])):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    
    return np.array([mean, error])


def montecarlo_integrate(func: callable, bounds: np.array):
    num_samples = 50000 #- d=2
    #num_samples = 150000  # d=3
    #num_samples = 100 ** bounds.shape[0]
    samples = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_samples, len(bounds)))
    values = func(samples.T)
    #values = func(samples)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    #return np.mean(values) * volume
    return jackknife(values) * volume


def get_corr_func_coord(cfgs: np.ndarray):
    """
    Return connected two-point correlation function (from distance)
    with errors for symmetric lattice along fixed axis (first).
    For the periodic boundary conditions we place the number of axis and the position
    of the initial cite does not matter.
    """
    mu = 1  # >=1
    corr_func = []
    if cfgs.ndim > 2:
        cfgs = np.mean(cfgs, axis=tuple(range(2, cfgs.ndim)))

    for shift in range(0, cfgs.shape[1]):
        corr_func.append(np.mean(cfgs * np.roll(cfgs, shift, mu), axis=0))

    shifted_cf = []
    for shift in range(0, cfgs.shape[1]):
        shifted_cf.append(np.roll(corr_func[shift], -shift, axis=0))

    shifted_cf = np.array(shifted_cf)

    return np.mean(shifted_cf, axis=1)



def get_corr_func_mom_parallel(cfgs: np.ndarray, p: np.ndarray):
    d = cfgs.ndim - 1
    L = cfgs.shape[1]
    samples_num = cfgs.shape[0] * L ** (d - 1)
    assert len(p) == L
    spatial_axis = tuple(np.arange(1, d + 1))

    shifts_coords = list(product(*[range(L)] * d))

    # Создаем разделяемую память для cfgs
    shm_cfgs = shared_memory.SharedMemory(create=True, size=cfgs.nbytes)
    shared_cfgs = np.ndarray(cfgs.shape, dtype=cfgs.dtype, buffer=shm_cfgs.buf)
    shared_cfgs[:] = cfgs[:]

    # Создаем разделяемую память для corrs
    corrs_shape = (samples_num, L)
    shm_corrs = shared_memory.SharedMemory(create=True, size=np.zeros(corrs_shape).nbytes)
    shared_corrs = np.ndarray(corrs_shape, dtype=np.float64, buffer=shm_corrs.buf)
    shared_corrs.fill(0)

    manager = Manager()
    lock = manager.Lock()

    def process_chunk(chunk, cfgs_shape, corrs_shape, shm_cfgs_name, shm_corrs_name):
        """Обрабатывает чанк сдвигов и обновляет corrs."""
        # Подключаемся к разделяемой памятиx
        existing_shm_cfgs = shared_memory.SharedMemory(name=shm_cfgs_name)
        existing_shm_corrs = shared_memory.SharedMemory(name=shm_corrs_name)

        local_cfgs = np.ndarray(cfgs_shape, dtype=np.float64, buffer=existing_shm_cfgs.buf)
        local_corrs = np.ndarray(corrs_shape, dtype=np.float64, buffer=existing_shm_corrs.buf)

        # Локальная сумма для текущего чанка
        local_chunk_corrs = np.zeros(corrs_shape)

        for shift in chunk:
            cos_values = np.cos(p @ np.array(shift))
            cos_values = cos_values.reshape((1,) * (local_cfgs.ndim - 1) + (-1,))
            local_chunk_corrs += (local_cfgs * np.roll(local_cfgs, shift, axis=spatial_axis) * cos_values).reshape(-1, L)

        # Синхронно обновляем общий массив corrs
        with lock:
            local_corrs += local_chunk_corrs
        # Закрываем память
        existing_shm_cfgs.close()
        existing_shm_corrs.close()

    # Разбиваем сдвиги на чанки
    num_chunks = 100
    chunk_size = len(shifts_coords) // num_chunks + 1
    chunks = [shifts_coords[i:i + chunk_size] for i in range(0, len(shifts_coords), chunk_size)]

    # Параллельная обработка чанков
    Parallel(n_jobs=6, backend="loky")(
        delayed(process_chunk)(
            chunk,
            cfgs.shape,
            corrs_shape,
            shm_cfgs.name,
            shm_corrs.name,
        )
        for chunk in tqdm(chunks)
    )

    # Закрываем и удаляем разделяемую память
    shm_cfgs.close()
    shm_cfgs.unlink()

    # Получаем итоговый corrs из общей памяти
    corrs = np.array(np.ndarray(corrs_shape, dtype=np.float64, buffer=shm_corrs.buf).T)

    shm_corrs.close()
    shm_corrs.unlink()

    # Расчет средних и ошибок через кросс-валидацию
    logger.info(f"Calculating means and error using cross-validation...")
    return np.array([cross_validation_mean_error_np(sample) for sample in corrs])

def get_corr_func_mom_optimized(cfgs: np.ndarray, p: np.ndarray):
    d = cfgs.ndim - 1
    L = cfgs.shape[1]
    samples_num = cfgs.shape[0] * L**(d-1)
    assert len(p) == L
    spatial_axis = tuple(np.arange(1, d + 1))

    shifts_coords = product(*[range(L)] * d)  #, total=L ** d)
    corrs = np.zeros((samples_num, L))
    ## TODO: брать одномерный массив shifts??
    for shift in tqdm(shifts_coords, total=L ** d):
        # tODO: проверить, что тут все хорошо и согласовано по размерностям
        cos_values = np.cos(p @ np.array(shift))
        cos_values = cos_values.reshape((1,) * (cfgs.ndim - 1) + (-1,))
        # готовим массив, чтобы потом просуммировать по сдвигам. Для одновременного учета всех импульсов используем векторизацию
        # также используем, что импульсов имеется одномерный массив, и все остальные измерения (0+все, кроме последнего пространственного)
        # дают нам просто большее количество выборок

        corrs += (cfgs * np.roll(cfgs, shift, axis=spatial_axis) * cos_values).reshape(-1, L)

    # останутся только разные выборки (N * L^d) + импульсы
    corrs = corrs.T
    # TODO: сразу сохранять фолды, а не весь массив, чтобы память поэкономить? пускай даже на 1000 элементов
    # TODO: через разделенную память разбить сдвиги на чанки и разделить между 2-3 процессорами
    logger.info(f"Calculating means and error using cross validation...")
    return np.array([cross_validation_mean_error_np(sample) for sample in corrs])


def compute_corr_for_shift(cfgs, shift_0, shift_1, p, L, d, spatial_axis):
    shift = np.concatenate((shift_0, [shift_1]))
    cos_values = (np.cos(p @ np.array(shift))).reshape((1,) * (cfgs.ndim - 1) + (-1,))
    return (cfgs * np.roll(cfgs, shift, axis=spatial_axis) * cos_values).reshape(-1, L)



def get_momenta_grid(M: int, d: int):
    """
    Функция для генерации одномерной (вдоль одной оси) сетки решеточных импульсов
     M - длина ребра куба в решетке. Важно точно попадать в импульсы, соответствующие решетке, иначе DFT будет оч сильно
     осциллировать относительно желаемого непрерывного результата.
    """
    assert d > 0
    return 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi





