import logging
from itertools import product

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)



def shuffled_group_jackknife_mean_error(samples: np.ndarray, k_folds: int = 100):
    """Return mean and estimated lower error bound using k-fold cross-validation."""
    samples = np.asarray(samples)

    n_samples = samples.shape[0]

    if not 2 <= k_folds <= n_samples:
        raise ValueError(
            f"k must satisfy 2 <= k <= n_samples, got {k_folds}"
        )

    np.random.shuffle(samples)

    total_sum = samples.sum(axis=0)

    means = []

    for fold in np.array_split(samples, k_folds, axis=0):
        means.append(
            (total_sum - fold.sum(axis=0))
            / (n_samples - len(fold))
        )

    means = np.asarray(means)

    mean = means.mean(axis=0)
    error = np.sqrt(k_folds) * means.std(axis=0, ddof=1)

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


def blocked_jackknife(
    samples: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block jackknife over consecutive retained HMC configurations.

    samples.shape == (n_samples, n_momenta)
    """
    n_samples = samples.shape[0]
    n_blocks = n_samples // block_size

    if n_blocks < 2:
        raise ValueError("At least two complete blocks are required.")

    n_used = n_blocks * block_size
    samples = samples[:n_used]

    block_sums = samples.reshape(
        n_blocks,
        block_size,
        samples.shape[1],
    ).sum(axis=1)

    total_sum = block_sums.sum(axis=0)

    leave_one_out = (
        total_sum[None, :] - block_sums
    ) / (n_used - block_size)

    mean = samples.mean(axis=0)

    jackknife_mean = leave_one_out.mean(axis=0)

    error = np.sqrt(
        (n_blocks - 1) / n_blocks
        * np.sum(
            (leave_one_out - jackknife_mean) ** 2,
            axis=0,
        )
    )

    return mean, error


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


def two_point_sample_fft(phi: np.ndarray) -> np.ndarray:
    """Compute D(p) for p=(2*pi*k/L,0,...,0), k=0,...,L-1."""
    transverse_axes = tuple(range(1, phi.ndim))

    projected = phi.sum(axis=transverse_axes)

    phi_p = np.fft.fft(projected)

    return np.abs(phi_p) ** 2 / phi.size


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
    return np.array([shuffled_group_jackknife_mean_error(sample) for sample in corrs])


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





