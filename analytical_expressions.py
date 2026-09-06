import numpy as np
from math import gamma as Gamma
from scipy.integrate import nquad
from core.utils import montecarlo_integrate, nquad_estimate
from core.error_propagator import Estimate

n = 2.  # 2 * n is a  power of interaction

# coefficients a dual action interaction expantion

a = Gamma(3. / 2 / n) * np.power(Gamma(2 * n + 1), 1 / n) / Gamma(1 / 2 / n)

b = ((3 * Gamma(3. / 2 / n) ** 2 - Gamma(1.0 / 2 / n) * Gamma(5.0 / 2 / n)) * np.power(Gamma(2 * n + 1), 2 / n)
     / Gamma(1. / 2 / n) ** 2)

c = (30 * Gamma(3. / 2 / n) ** 3 - 15 * Gamma(1. / 2 / n) * Gamma(5. / 2 / n) * Gamma(3. / 2 / n) +
     Gamma(1. / 2 / n) ** 2 * Gamma(7. / 2 / n)) * np.power(Gamma(2 * n + 1), 3 / n) / Gamma(1. / 2 / n) ** 3

# distinguish from dimension!!!!
dd = 288.362

r = 8497.59  # 10th order of dual potential expansion


def G_xi_w(alpha: float, gamma: float, xi: np.ndarray | list) -> np.ndarray:
    """ 
        Weak expansion free Green's function 
    """
    return 1. / (4 * alpha * np.sum(np.sin(xi / 2) ** 2, axis=0) + gamma)

def G_xi_s(alpha: float, gamma: float, g: float, xi: np.ndarray) -> np.ndarray:
    """ 
        Strong expansion free Green's function 
    """
    s = np.sum(np.sin(xi / 2) ** 2, axis=0)
    return (g ** 2 / a) * (4 * alpha * s + gamma) / (4 * alpha * s + gamma + g ** 2 / a)


def G_0_w(alpha: float, gamma: float, d: int) -> Estimate:
    """ 
        Weak expansion one-loop integral G_0
    """
    return nquad_estimate(lambda *xi: G_xi_w(alpha, gamma, np.array(xi)), [[0, 2 * np.pi] for _ in range(d)]) / ((2 * np.pi) ** d)


def G_0_s(alpha: float, gamma: float, g: float, d: int) -> Estimate:
    """ 
        Strong expansion one-loop integral G_0
    """
    return nquad_estimate(lambda *xi: G_xi_s(alpha, gamma, g, np.array(xi)), [[0, 2 * np.pi] for _ in range(d)]) / ((2 * np.pi) ** d)


def triple_product(G: callable, d, *args, **kwargs) -> float:
    """
        Shorthand nonation for product with momentum conservation with given (Green's) function G
    """
    l1 = args[:d]
    l2 = args[d:2 * d]
    l3 = args[2 * d:3 * d]
    l4 = [l1[i] + l2[i] + l3[i] for i in range(d)]
    return G(**kwargs, xi=l1) * G(**kwargs, xi=l2) * G(**kwargs, xi=l3) * G(**kwargs, xi=l4)


def f_w(alpha: float, gamma: float, g: float, d: int) -> np.array:
    """
        Weak coupling expansion of free energy per site
    """

    f1 = g ** 4 / 8. * G_0_w(alpha, gamma, d) ** 2
    #f_2_1 = (- g ** 8 / 16. / (2 * np.pi) ** (3 * d) * G_0_w(alpha, gamma, d) ** 2 *
    #         montecarlo_integrate(lambda xi: G_xi_w(alpha, gamma, np.array(xi))**2,
    #               np.array([[0, 2 * np.pi] for _ in range(d)])))


    f_2_1 = (- g ** 8 / 16. / (2 * np.pi) ** d * G_0_w(alpha, gamma, d) ** 2 *
             nquad_estimate(lambda *xi: G_xi_w(alpha, gamma, np.array(xi))**2,
                   [[0, 2 * np.pi] for _ in range(d)]))


    f_2_2 = (- g ** 8 / 24. / 2. / (2 * np.pi) ** (3 * d) *
             montecarlo_integrate(lambda args: G_xi_w(alpha, gamma, args[:d, :]) *
                                               G_xi_w(alpha, gamma, args[d:2 * d, :]) *
                                               G_xi_w(alpha, gamma, args[2 * d:3 * d, :]) *
                                               G_xi_w(alpha, gamma, args[: d, :] + args[d: 2 * d, :] + args[2 * d:3 * d, :]),
                                  np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    f_est = f1 + f_2_1 + f_2_2

    return np.array([f_est.value, f_est.error])


def f_s(alpha: float, gamma: float, g: float, d: int) -> np.array:
    """
        Strong coupling expansion of free energy per site
    """
    f_0 = ( Estimate(1.0, 0.0) * (0.5 * np.log(2 * np.pi) -
                                  np.log(Gamma(1. / 2 / n) * np.power(Gamma(2 * n + 1), 1. / 2 / n) / n)) + 0.5 / ((2 * np.pi) ** d) *
            np.array(
                nquad_estimate(lambda *xi: np.log(g ** 2 / G_xi_s(alpha, gamma, g, np.array(xi))),
                      [[0, 2 * np.pi] for _ in range(d)])
            )
            )

    f_1 = b / g ** 4 / 8. * G_0_s(alpha, gamma, g, d) ** 2

    # f_1 = Estimate(0.0, 0.0)

    f_2 = c / g ** 6 / 48. * G_0_s(alpha, gamma, g, d) ** 3

    # f_2 = Estimate(0.0, 0.0)

    f_4_1 = dd / g ** 8 / 384. * G_0_s(alpha, gamma, g, d) ** 4
    # f_4_1 = Estimate(0., 0.)

    f_4_2 = (- b ** 2 / g ** 8 / 16. / (2 * np.pi) ** d * G_0_s(alpha, gamma, g, d) ** 2 *
             nquad_estimate(lambda *xi: G_xi_s(alpha, gamma, g, np.array(xi)) ** 2,
                   [[0, 2 * np.pi] for _ in range(d)]))

    # f_4_2 = Estimate(0., 0.)
    
    f_4_3 = (- b ** 2 / 24. / 2.0 / g ** 8 / (2 * np.pi) ** (3 * d) *
                        montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[2 * d:3 * d, :]) *
                                                          G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + args[2 * d: 3 * d, :]),
                                             np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    # f_4_3 = Estimate(0., 0.)

    f_est: Estimate = f_0 + f_1 + f_2 + f_4_1 + f_4_2 + f_4_3

    return np.array([f_est.value, f_est.error])


def two_point_correlator_amputated_w(alpha: float, gamma: float, g: float, d: int, xi: np.array) -> np.array:
    """
        Weak coupling expansion of (amputated) two-poing function
    """

    G_2_0 = Estimate(1.0, 0.0) / G_xi_w(alpha, gamma, xi) 
    G_2_1 = - 0.5 * (g ** 4) * G_0_w(alpha, gamma, d)
    G_2_2_1 = 0.25 * (g ** 8) * (G_0_w(alpha, gamma, d) ** 2) * G_xi_w(alpha, gamma, xi)
    G_2_2_2 = (0.25 * (g ** 8) / ((2 * np.pi) ** d) * G_0_w(alpha, gamma, d) *
               nquad_estimate(lambda *zeta: G_xi_w(alpha, gamma, np.array(zeta))**2,
                     [[0, 2 * np.pi] for _ in range(d)]))

    G_2_2_3 = (1. / 6 * g ** 8 / ((2 * np.pi) ** (2 * d)) *
               montecarlo_integrate(lambda args: G_xi_w(alpha, gamma, args[:d, :]) *
                                                 G_xi_w(alpha, gamma, args[d:2 * d, :]) *
                                                 G_xi_w(alpha, gamma, args[: d, :] + args[d: 2 * d, :] + np.asarray(xi)[:, None]),
                                    np.array([[0, 2 * np.pi] for _ in range(2 * d)])))

    G_2_est = G_2_0 + G_2_1 + G_2_2_1 + G_2_2_2 + G_2_2_3

    return np.array([G_2_est.value, G_2_est.error])


def two_point_correlator_amputated_s(xi: np.array, alpha: float, gamma: float, g: float, d: int) -> np.array:
    """
        Strong coupling expansion of (amputated) two-poing function
    """

    I2 = nquad_estimate(lambda *zeta: G_xi_s(alpha, gamma, g, np.array(zeta))**2,
                                  [[0, 2 * np.pi] for _ in range(d)])

    I3 = montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                     G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                     G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + np.asarray(xi)[:, None]),
                                        np.array([[0, 2 * np.pi] for _ in range(2 * d)]))
    
    G_2_0 = Estimate(1.0, 0.0) / G_xi_s(alpha, gamma, g, xi)
    G_2_1 = - 0.5 * b / (g ** 4) * G_0_s(alpha, gamma, g, d)

    G_2_2 = - 1. / 8 * c / (g ** 6) * (G_0_s(alpha, gamma, g, d) ** 2)

    G_2_3_1 = 0.25 * b**2 / (g ** 8) * (G_0_s(alpha, gamma, g, d) ** 2) * G_xi_s(alpha, gamma, g, xi)  # 1-particle reductible!

    G_2_3_2 = (0.25 * b**2 / (g ** 8) / ((2 * np.pi) ** d) * G_0_s(alpha, gamma, g, d) * I2)

    G_2_3_3 = (1. / 6 * b**2 / g ** 8 / ((2 * np.pi) ** (2 * d)) * I3)

    G_2_3_4 = - 1. / 6 / 8 * dd / (g**8) * (G_0_s(alpha, gamma, g, d) ** 3)


    G_2_4_1 = -1. / 16 / 24 * r / (g**10) * (G_0_s(alpha, gamma, g, d) ** 4)

    G_2_4_2 = 1. / 8 * b * c / (g**10) * (G_0_s(alpha, gamma, g, d) ** 3) * G_xi_s(alpha, gamma, g, xi)  # 1-particle reductible!

    # SUM OF 2 DIAGRAMS WITH 2 BRIDGES BETWEEN 6 AND 4 VERTICES
    G_2_4_3 = ( 
        (1.0 / 8 + 1.0 / 16) * b * c  / (g**10) * (G_0_s(alpha, gamma, g, d) ** 2) * 1 / ((2 * np.pi) **d) * I2
    )

    G_2_4_4 = (1.0 / 6 * b * c / (g**10) * G_0_s(alpha, gamma, g, d) / ((2 * np.pi) ** (2 * d)) * I3)

    G_2_4_5 = (1.0 / 24 * b * c / (g**10) / ((2 * np.pi) ** (3 * d)) *
                montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[2 * d:3 * d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + args[2 * d: 3 * d, :]),
                                                 np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    G_2_est = G_2_0 + G_2_1 + G_2_2 + G_2_3_1 + G_2_3_2 + G_2_3_3 + G_2_3_4 + G_2_4_1 + G_2_4_2 + G_2_4_3 + G_2_4_4 + G_2_4_5

    return np.array([G_2_est.value, G_2_est.error])


def two_point_correlator_s_dyson(xi: np.array, alpha: float, gamma: float, g: float, d: int) -> np.array:
    """
        Strong coupling expansion of (amputated) two-poing function
    """


    I2 = nquad_estimate(lambda *zeta: G_xi_s(alpha, gamma, g, np.array(zeta))**2,
                                  [[0, 2 * np.pi] for _ in range(d)])

    I3 = montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                     G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                     G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + np.asarray(xi)[:, None]),
                                        np.array([[0, 2 * np.pi] for _ in range(2 * d)]))
    
    G_2_0 = Estimate(1.0, 0.0) / G_xi_s(alpha, gamma, g, xi)

    G_2_1 = - 0.5 * b / (g ** 4) * G_0_s(alpha, gamma, g, d)
    
    G_2_2 = - 1. / 8 * c / (g ** 6) * (G_0_s(alpha, gamma, g, d) ** 2)

    G_2_3_2 = (0.25 * b**2 / (g ** 8) / ((2 * np.pi) ** d) * G_0_s(alpha, gamma, g, d) * I2)

    G_2_3_3 = (1. / 6 * b**2 / g ** 8 / ((2 * np.pi) ** (2 * d)) * I3)

    G_2_3_4 = - 1. / 6 / 8 * dd / (g**8) * (G_0_s(alpha, gamma, g, d) ** 3)

    G_2_4_1 = -1. / 16 / 24 * r / (g**10) * (G_0_s(alpha, gamma, g, d) ** 4)

    # SUM OF 2 DIAGRAMS WITH 2 BRIDGES BETWEEN 6 AND 4 VERTICES
    G_2_4_3 = ( 
        (1.0 / 8 + 1.0 / 16) * b * c  / (g**10) * (G_0_s(alpha, gamma, g, d) ** 2) * 1 / ((2 * np.pi) **d) * I2
    )

    G_2_4_4 = (1.0 / 6 * b * c / (g**10) * G_0_s(alpha, gamma, g, d) / ((2 * np.pi) ** (2 * d)) * I3)

    G_2_4_5 = (1.0 / 24 * b * c / (g**10) / ((2 * np.pi) ** (3 * d)) *
                montecarlo_integrate(lambda args: G_xi_s(alpha, gamma, g, args[:d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[d:2 * d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[2 * d:3 * d, :]) *
                                                    G_xi_s(alpha, gamma, g, args[: d, :] + args[d: 2 * d, :] + args[2 * d: 3 * d, :]),
                                                 np.array([[0, 2 * np.pi] for _ in range(3 * d)])))

    Sigma_est = -(G_2_1 + G_2_2  + G_2_3_2 + G_2_3_3 + G_2_3_4 + G_2_4_1 + G_2_4_3 + G_2_4_4 + G_2_4_5)  # self-energy up to fourth order in 1/g^2


    with open("values.txt", "a") as f:
        f.write(f"G_2_1 G_2_2 G_2_3_2 G_2_3_3 G_2_3_4 G_2_4_1 G_2_4_3 G_2_4_4 G_2_4_5 Sigma_est (xi={xi}, g={g})\n")
        f.write(
            f"{G_2_1} {G_2_2} {G_2_3_2} {G_2_3_3} {G_2_3_4} {G_2_4_1} {G_2_4_3} {G_2_4_4} {G_2_4_5} {Sigma_est} \n"
        )

    G_2_est = 1.0 / (G_2_0 + Sigma_est)

    return np.array([G_2_est.value, G_2_est.error])

