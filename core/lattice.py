import numpy as np
import copy

# TODO: почекать через cp, мб будет быстрее, чем np

class Lattice:
    def __init__(self, M, d, alpha, gamma, G):
        self.M = M
        self.d = d
        self.shape = [M for _ in range(d)]
        self.G = G
        self.alpha = alpha
        self.gamma = gamma

        # добавим регуляризационный фактор, отрескейлит поле, чтобы метод лучше сходился. На средние не повлияет.

        # инициализируем поле случайной конфигурацией и считаем на ней действие
        self.phi = np.random.randn(*self.shape)
        self.action = self.get_action()

    def get_action(self):
        kinetic_part = 0.5 * (self.gamma + 2 * self.d * self.alpha) * (self.phi ** 2)

        # прибавляем сдвиговые слагаемые в лапласиане + используем периодичность краевых условий
        for mu in range(self.d):
            kinetic_part -= self.alpha * self.phi * np.roll(self.phi, 1, mu)
            #kinetic_part -= 0.5 * self.alpha * self.phi * np.roll(self.phi, -1, mu)

        interaction_part = self.G / 24. * self.phi ** 4
        action = interaction_part + kinetic_part

        return action.sum()

    def get_action_gradient(self):
        """
        функция для получения смещения полевых переменных в методе hmc
        """

        # начинаем с простых слагаемых, зависящих от значения только в одном узле
        gradient = (self.gamma + 2 * self.d * self.alpha) * self.phi + self.G * ( (self.phi ** 3) / 6.)

        for mu in range(self.d):
            gradient -= self.alpha * (np.roll(self.phi, 1, mu) + np.roll(self.phi, -1, mu))

        return gradient

    def get_hamiltonian(self, chi, action):
        return 0.5 * np.sum(chi ** 2) + action

    def hmc(self, n_steps=100):

        """
        функция для осуществления шагов Hybrid Monte-Carlo.  chi - импульсы, phi - поля. Согласно уравнениям,
        обновляем phi на импульсы, а chi - на минус-вариацию действия, все умножаем на временной шаг.

        На последнем шаге какой-то подгон с обновлением импульсов только на половинку, ну да ладно.
        """

        dt = 1 / n_steps
        phi_0 = copy.deepcopy(self.phi)
        # TODO: проверить, что будет, если брать распределение по гауссу, как советуют, а не равномерное
        chi = np.random.randn(*self.shape)

        #chi = np.random.multivariate_normal(
        #    np.zeros(self.M ** self.d),
        #    np.identity(self.M ** self.d),
        #).reshape(self.shape)

        S_0 = self.get_action()
        H_0 = self.get_hamiltonian(chi, S_0)

        chi -= 0.5 * dt * self.get_action_gradient()

        for i in range(n_steps - 1):
            self.phi += dt * chi
            chi -= dt * self.get_action_gradient()
        self.phi += dt * chi
        chi -= 0.5 * dt * self.get_action_gradient()

        self.action = self.get_action()
        dH = self.get_hamiltonian(chi, self.action) - H_0

        if dH > 0:
            # в этом случае принимаем изменения с вероятностью e^{-dH}
            if np.random.rand() >= np.exp(-dH):
                #print(dH)
                self.phi = phi_0
                self.action = S_0
                return phi_0, False

        # в этом случае точно одобряем изменение
        return self.phi, True
