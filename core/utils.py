import numpy as np
from numba import jit
from tqdm import tqdm


def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
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
        #correlator.append(np.mean(np.mean(cfgs * np.roll(cfgs, shift, axis=spatial_axis), axis=0)))
        correlator.append(np.mean(cfgs * np.roll(cfgs, shift, axis=spatial_axis), axis=spatial_axis))
        #break

    correlator = np.array(correlator)

    #return np.sum(correlator * np.cos(p @ shifts_coords.T), axis=1)

    corrs = np.sum(np.expand_dims(correlator, axis=0) * np.expand_dims(np.cos(p @ shifts_coords.T), axis=2),
                   axis=1)

    return np.array([jackknife(sample) for sample in corrs])

def get_momenta_grid(M: int, d: int):
    """
    Функция для генерации одномерной (вдоль одной оси) сетки решеточных импульсов
     M - длина ребра куба в решетке. Важно точно попадать в импульсы, соответствующие решетке, иначе DFT будет оч сильно
     осциллировать относительно желаемого непрерывного результата.
    """
    assert d > 0
    return 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi




