from core.lattice import Lattice
from core.functions import get_mag, get_chi2, get_abs_mag, get_corr_func
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt


N = 32
d = 3
alpha = - 0.3
gamma = 0.3
g = 0.1

L = Lattice(N, d, alpha, gamma, g)

cfgs = []

# разогреваем метод, чтобы получить правильную начальную конфигурацию, их пока не записываем в cfgs
print("Starting heating..")
for _ in tqdm(range(1000)):
    phi, accepted = L.hmc()

print("Heating finished! \n")

accepted_num = 0

print("Starting computations..")
n_iter = 10000

for i in tqdm(range(n_iter)):
    phi, accepted = L.hmc()
    if accepted:
        accepted_num += 1

    if i % 10 == 0:
        cfgs.append(copy.deepcopy(phi))

print("Computations finished!")

print(f"Acceptance rate: {accepted_num/n_iter}")

cfgs = np.array(cfgs)

np.save('lattice-phi4', cfgs)

print(f"{np.shape(cfgs)[0]} saved!")
print(plt.imshow(cfgs[-1]))

M, M_err = get_mag(cfgs)
M_abs, M_abs_err = get_abs_mag(cfgs)
chi2, chi2_err = get_chi2(cfgs)

print("M = %.4f +/- %.4f" % (M, M_err))
print("|M| = %.4f +/- %.4f" % (M_abs, M_abs_err))
print("chi2 = %.4f +/- %.4f" % (chi2, chi2_err))

corr_func = get_corr_func(cfgs)

fig, ax = plt.subplots(1,1, dpi=125, figsize=(8, 8))
plt.xticks([i for i in range(1, N, 4)])
ax.errorbar(corr_func[:, 0], corr_func[:,1], yerr=corr_func[:, 2], label='2-point correlator')
plt.legend(prop={'size': 16})
plt.show()
