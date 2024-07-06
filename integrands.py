import numpy as np
import scipy as sp
from math import gamma as Gamma
from scipy.integrate import nquad
from core.utils import montecarlo_integrate

n = 2.

a = Gamma(3. / 2 / n) * np.power(Gamma(2 * n + 1), 1 / n) / Gamma(1 / 2 / n)

b = ((3 * Gamma(3. / 2 / n) ** 2 - Gamma(1 / 2 / n) * Gamma(5 / 2 / n)) * np.power(Gamma(2 * n + 1), 2 / n)
     / Gamma(1 / 2 / n) ** 2)

c = (30 * Gamma(3. / 2 / n) ** 3 - 15 * Gamma(1 / 2 / n) * Gamma(5 / 2 / n) * Gamma(3 / 2 / n) +
     Gamma(1 / 2 / n) ** 2 * Gamma(7 / 2 / n)) * np.power(Gamma(2 * n + 1), 3 / n) / Gamma(1 / 2 / n) ** 3

# distinguish from dimension!!!!
dd = 288.362


def G_xi_w(alpha: float, gamma: float, xi: np.ndarray | list) -> np.ndarray:
    #axis = xi.ndim
    return 1. / (4 * alpha * np.sum(np.sin(xi / 2) ** 2, axis=0) + gamma)

def G_xi_s(alpha: float, gamma: float, g: float, xi: np.ndarray) -> np.ndarray:
    #axis = xi.ndim
    s = np.sum(np.sin(xi / 2) ** 2, axis=0)
    return (g ** 2 / a) * (4 * alpha * s + gamma) / (4 * alpha * s + gamma + g ** 2 / a)


def G_0_w(alpha: float, gamma: float, d: int) -> np.ndarray:
    #return np.array(montecarlo_integrate(lambda xi: G_xi_w(alpha, gamma, np.array(xi)),
    #                                     np.array([[0, 2 * np.pi] for _ in range(d)])))
    return np.array(nquad(lambda *xi: G_xi_w(alpha, gamma, np.array(xi)), [[0, 2 * np.pi] for _ in range(d)]))


def G_0_s(alpha: float, gamma: float, g: float, d: int) -> np.ndarray:
    #return np.array(montecarlo_integrate(lambda xi: G_xi_s(alpha, gamma, g, np.array(xi)),
    #                                     np.array([[0, 2 * np.pi] for _ in range(d)])))
    return np.array(nquad(lambda *xi: G_xi_s(alpha, gamma, g, np.array(xi)), [[0, 2 * np.pi] for _ in range(d)]))

def triple_product(G: callable, d, *args, **kwargs) -> float:
    l1 = args[:d]
    l2 = args[d:2 * d]
    l3 = args[2 * d:3 * d]
    l4 = [l1[i] + l2[i] + l3[i] for i in range(d)]
    return G(**kwargs, xi=l1) * G(**kwargs, xi=l2) * G(**kwargs, xi=l3) * G(**kwargs, xi=l4)


def f_w(alpha: float, gamma: float, g: float, d: int) -> np.array:
    f1 = g ** 4 / 8. / (2 * np.pi) ** (2 * d) * G_0_w(alpha, gamma, d) ** 2
    #f_2_1 = (- g ** 8 / 16. / (2 * np.pi) ** (3 * d) * G_0_w(alpha, gamma, d) ** 2 *
    #         montecarlo_integrate(lambda xi: G_xi_w(alpha, gamma, np.array(xi))**2,
    #               np.array([[0, 2 * np.pi] for _ in range(d)])))


    f_2_1 = (- g ** 8 / 16. / (2 * np.pi) ** (3 * d) * G_0_w(alpha, gamma, d) ** 2 *
             nquad(lambda *xi: G_xi_w(alpha, gamma, np.array(xi))**2,
                   [[0, 2 * np.pi] for _ in range(d)]))


    f_2_2 = (- g ** 8 / 24. / (2 * np.pi) ** (3 * d) *
             montecarlo_integrate(lambda args: G_xi_w(alpha, gamma, args[:d, :]) *
                                               G_xi_w(alpha, gamma, args[d:2 * d, :]) *
                                               G_xi_w(alpha, gamma, args[2 * d:3 * d, :]) *
                                               G_xi_w(alpha, gamma, args[: d, :] + args[d: 2 * d, :] + args[2 * d:3 * d, :]),
                                  np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    return f1 + f_2_1 + f_2_2


def f_s(alpha: float, gamma: float, g: float, d: int) -> np.array:
    f_0, f_0_err = ( np.array([1., 0.]) * (0.5 * np.log(2 * np.pi) -
                                  np.log(Gamma(1. / 2 / n) * np.power(Gamma(2 * n + 1), 1. / 2 / n) / n)) + 0.5 / ((2 * np.pi) ** d) *
            np.array(
                nquad(lambda *xi: np.log(g ** 2 / G_xi_s(alpha, gamma, g, np.array(xi))),
                      [[0, 2 * np.pi] for _ in range(d)])
            )
            )

    f_1, f_1_err = b / g ** 4 / 8. / (2 * np.pi) ** (2 * d) * G_0_s(alpha, gamma, g, d) ** 2

    f_2, f_2_err = c / g ** 6 / 48. / (2 * np.pi) ** (3 * d) * G_0_s(alpha, gamma, g, d) ** 3

    f_4_1, f_4_1_err = dd / g ** 8 / 384. / (2 * np.pi) ** (4 * d) * G_0_s(alpha, gamma, g, d) ** 4

    #f_4_2 = (- b ** 2 / g ** 8 / 16. / (2 * np.pi) ** (3 * d) * G_0_s(alpha, gamma, g, d) ** 2 *
    #         montecarlo_integrate(lambda xi: G_xi_s(alpha, gamma, g, np.array(xi)) ** 2,
    #               np.array([[0, 2 * np.pi] for _ in range(d)])))

    f_4_2, f_4_2_err = (- b ** 2 / g ** 8 / 16. / (2 * np.pi) ** (3 * d) * G_0_s(alpha, gamma, g, d) ** 2 *
             nquad(lambda *xi: G_xi_s(alpha, gamma, g, np.array(xi)) ** 2,
                   [[0, 2 * np.pi] for _ in range(d)]))

    """    
    f_4_2 = (- b ** 2 / 24. / (2 * np.pi) ** (3 * d) *
             np.array(nquad(lambda *args: G_xi_s(alpha, gamma, g, np.array(args[:d])) *
                                          G_xi_s(alpha, gamma, g, np.array(args[d:2*d])) *
                                          G_xi_s(alpha, gamma, g, np.array(args[2*d:3*d])) *
                                          G_xi_s(alpha, gamma, g,  np.array(args[:d]) + np.array(args[d:2*d]) + np.array(args[2*d:3*d])),
                            [[0, 2 * np.pi] for _ in range(3 * d)])))
    """

    f_4_3, f_4_3_err = (- b ** 2 / 24. / g ** 8 / (2 * np.pi) ** (3 * d) *
                        montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[2 * d:3 * d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + args[2 * d: 3 * d, :]),
                                             np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    return (f_0 + f_1 + f_2 + f_4_1 + f_4_2 + f_4_3,
            np.sum(np.abs(np.array([f_0_err, f_1_err, f_2_err, f_4_1_err, f_4_2_err, f_4_3_err]))))


def two_point_correlator_amputated_w(alpha: float, gamma: float, g: float, d: int, xi: np.array) -> np.array:
    G_2_0 = np.array([1., 0.]) / G_xi_w(alpha, gamma, xi)
    G_2_1 = - 0.5 * (g ** 4) * G_0_w(alpha, gamma, d) / (2 * np.pi) ** d
    G_2_2_1 = 0.125 * (g ** 8) / ((2 * np.pi) ** (3 * d)) * (G_0_w(alpha, gamma, d) ** 2) * G_xi_w(alpha, gamma, xi)
    G_2_2_2 = (0.125 * (g ** 8) / ((2 * np.pi) ** (3 * d)) * G_0_w(alpha, gamma, d) *
               nquad(lambda *zeta: G_xi_w(alpha, gamma, np.array(zeta))**2,
                     [[0, 2 * np.pi] for _ in range(d)]))

    G_2_2_3 = (1. / 6 * g ** 8 / ((2 * np.pi) ** (3 * d)) *
               montecarlo_integrate(lambda args: G_xi_w(alpha, gamma, args[:d, :]) *
                                                 G_xi_w(alpha, gamma, args[d:2 * d, :]) *
                                                 G_xi_w(alpha, gamma, args[: d, :] + args[d: 2 * d, :]),
                                    np.array([[0, 2 * np.pi] for _ in range(2 * d)])))

    return G_2_0 + G_2_1 + G_2_2_1 + G_2_2_2 + G_2_2_3



def two_point_correlator_amputated_s(xi: np.array, alpha: float, gamma: float, g: float, d: int) -> np.array:
    G_2_0 = np.array([1., 0.]) / G_xi_s(alpha, gamma, g, xi)
    G_2_1 = - 0.5 * b / (g ** 4) * G_0_s(alpha, gamma, g, d) / (2 * np.pi) ** d

    G_2_2 = - 1. / 8 * c / (g ** 6) * (G_0_s(alpha, gamma, g, d) ** 2) / (2 * np.pi) ** (2 * d)

    G_2_3_1 = 0.125 * b**2 / (g ** 8) / ((2 * np.pi) ** (3*d)) * (G_0_s(alpha, gamma, g, d) ** 2) * G_xi_s(alpha, gamma, g, xi)

    G_2_3_2 = (0.125 * b**2 / (g ** 8) / ((2 * np.pi) ** (2 * d)) *
               np.array(nquad(lambda *zeta: G_xi_s(alpha, gamma, g, np.array(zeta))**2,
                              [[0, 2 * np.pi] for _ in range(d)])))

    G_2_3_3 = (1. / 6 * b**2 / g ** 8 / ((2 * np.pi) ** (3 * d)) *
               montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                 G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                 G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :]),
                                    np.array([[0, 2 * np.pi] for _ in range(2 * d)])))

    G_2_3_4 = - 1. / 6 / 8 * dd / (g**8) / ((2 * np.pi) ** (3 * d)) * (G_0_s(alpha, gamma, g, d) ** 3)

    return G_2_0 + G_2_1 + G_2_2 + G_2_3_1 + G_2_3_2 + G_2_3_3 + G_2_3_4
