import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from integrands import f_s, f_w

alpha = 1.
gamma = 1.
d = 3

f_num = pd.read_csv(f'data/free_energy_{d}.csv')

g_grid_w = np.linspace(0.0, 2.5, 150)
G_grid_w = g_grid_w ** 4

#G_grid_w = np.linspace(0.0, 16., 100)
#g_grid_w = G_grid_w ** 0.25

g_grid_s = np.linspace(1, 2.8, 20)
G_grid_s = g_grid_s ** 4

#f_w_grid = np.array([f_w(alpha, gamma, g, d) for g in tqdm(g_grid_w)]).T
f_s_grid = np.array([f_s(alpha, gamma, g, d) for g in tqdm(g_grid_s)]).T

print(f_s_grid)

plt.figure(figsize=(10,8))

plt.plot(G_grid_s, np.abs(f_s_grid[1]), label='Strong Coupling series')
plt.scatter(f_num["g^4"], f_num["f_error"], label='Numerical simulation', color="red")

plt.xlabel(r"$g^4$", fontsize=20)
plt.ylabel(r"$f$", fontsize=20, rotation=0)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim((1., 50.))
#plt.ylim((0., .120))
plt.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.grid()
plt.savefig(f"img/err_f(g)_{d}.png")

plt.show()

