import numpy as np
from tqdm import tqdm
#import numba
from numba import njit
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

logger = logging.getLogger(__name__)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.INFO)


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


def get_corr_func_mom(cfgs: np.ndarray, p: np.ndarray):
    correlator = []
    d = cfgs.ndim - 1
    spatial_axis = tuple(np.arange(1, d + 1))

    shifts_coords = np.meshgrid(*[np.arange(0, cfgs.shape[1])] * d, indexing="ij")
    shifts_coords = np.array([coord.flatten() for coord in shifts_coords]).T
    for shift in tqdm(shifts_coords):
        #corr_func.append(np.roll(np.mean(cfgs * np.roll(cfgs, shift, axis=spatial_axis), axis=0), -shift,
        #                         axis=(spatial_axis - 1)))
        correlator.append(np.mean(np.mean(cfgs * np.roll(cfgs, shift, axis=spatial_axis), axis=0)))
        #correlator.append(np.mean(cfgs * np.roll(cfgs, shift, axis=spatial_axis), axis=spatial_axis))

    correlator = np.array(correlator)

    #return np.sum(correlator * np.cos(p @ shifts_coords.T), axis=1)

    corrs = np.sum(np.expand_dims(correlator, axis=0) * np.expand_dims(np.cos(p @ shifts_coords.T), axis=2),
                   axis=1)

    return np.array([jackknife(sample) for sample in corrs])


@njit
def process_shifts(cfgs, p, samples_num, spatial_axis, d, L):
    corrs = np.zeros((samples_num, L))

    # Перебираем возможные сдвиги по каждому измерению без использования product.
    for shift_indices in range(L ** d):
        if shift_indices % 100 == 0:
            print(shift_indices)
        # Инициализируем массив сдвигов как int
        shift = np.empty(d, dtype=np.int64)
        index = shift_indices
        for i in range(d):
            shift[i] = index % L
            index //= L

        # Преобразуем shift в float64 непосредственно перед операцией умножения
        cos_values = np.cos(p @ shift.astype(np.float64))

        # Применяем np.roll последовательно по каждой оси
        rolled_cfgs = cfgs.copy()
        for i in range(d):
            rolled_cfgs = np.roll(rolled_cfgs, shift[i], axis=spatial_axis[i])

        # Обновляем corrs, учитывая сдвиг, косинус и векторизацию
        corrs += (rolled_cfgs * np.expand_dims(cos_values, axis=tuple(np.arange(0, d + 1)))).reshape(-1, L)

    return corrs

#@jit(nopython=True, parallel=True, fastmath=True)


def get_corr_func_mom_optimized(cfgs: np.ndarray, p: np.ndarray):
    d = cfgs.ndim - 1
    L = cfgs.shape[1]
    samples_num = cfgs.shape[0] * L**(d-1)
    assert len(p) == L
    spatial_axis = tuple(np.arange(1, d + 1))

    # Генерируем сдвиги на лету с помощью product и tqdm
    shifts_coords = product(*[range(L)] * d) #, total=L ** d)
    corrs = np.zeros((samples_num, L))
    ## TODO: брать одномерный массив shifts??
    for i, shift in tqdm(enumerate(shifts_coords), total=L ** d):
        cos_values = np.cos(p @ np.array(shift))
        cos_values = cos_values.reshape((1,) * (cfgs.ndim - 1) + (-1,))
        # готовим массив, чтобы потом просуммировать по сдвигам. Для одновременного учета всех импульсов используем векторизацию
        # также используем, что импульсов имеется одномерный массив, и все остальные измерения (0+все, кроме последнего пространственного)
        # дают нам просто большее количество выборок

        #rolled_cfgs = np.roll(cfgs, shift, axis=spatial_axis)
        #mult_result = cfgs * rolled_cfgs
        #mult_result *= cos_values
        #corrs += mult_result.reshape(-1, L)''

        corrs += (cfgs * np.roll(cfgs, shift, axis=spatial_axis) * cos_values).reshape(-1, L)

    logger.info(f"Taking sum over all shifts...")  # останутся только разные выборки (N * L^d) + импульсы
    corrs = corrs.T
    logger.info(f"Calculating means and error using cross validation...")
    return np.array([cross_validation_mean_error_np(sample) for sample in corrs])
    #return np.array([jackknife(sample) for sample in grouped_data])



def get_momenta_grid(M: int, d: int):
    """
    Функция для генерации одномерной (вдоль одной оси) сетки решеточных импульсов
     M - длина ребра куба в решетке. Важно точно попадать в импульсы, соответствующие решетке, иначе DFT будет оч сильно
     осциллировать относительно желаемого непрерывного результата.
    """
    assert d > 0
    return 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi





