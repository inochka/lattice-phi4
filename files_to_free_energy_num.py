import numpy
import numpy as np
import os
import re
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

results_list = []

d = 2

for entry in os.scandir("./data"):
    search_res = re.search(rf"{d}_(.*)_(.*)\.npy", entry.name)
    if search_res:
        print(f"Processing file {entry.name}...")
        groups = search_res.groups()
        arr = np.load(entry.path)
        av_4_point = numpy.mean(arr ** 4)
        results_list.append({"g^4": float(groups[0]), "gamma": float(groups[1]), "<phi^4>": av_4_point})

results = pd.DataFrame(results_list, columns=["g^4", "gamma", "<phi^4>"])

print(results.head())
print(len(results))

f_derivatives = []

f_results_list = []

G_max = np.max(results["g^4"])

for gamma in set(results["gamma"]):
    res_gamma = results.query(f"gamma=={gamma}")
    f_derivative = interp1d(res_gamma["g^4"], res_gamma["<phi^4>"], kind='cubic', fill_value="extrapolate")
    print(f_derivative(np.linspace(0., G_max, 50)))

    def integrand(x):
        return f_derivative(x) # / 6 * (x ** 3)

    for G in np.sort(res_gamma["g^4"]):
        res_tuple = quad(integrand, 0., G)
        f_results_list.append({"f": res_tuple[0], "f_error": res_tuple[1], "g^4": G, "gamma": gamma})

f_results = pd.DataFrame(f_results_list, columns=["g^4", "gamma", "f", "f_error"])

print(f_results.query("gamma==1").head())
print(len(f_results))

f_res_1 = f_results.query("gamma==1")
f_res_1["f"] = f_res_1["f"] / 24

plt.figure(figsize=(10,8))
print(list(np.power(f_res_1["g^4"], 0.25)))
plt.plot(f_res_1["g^4"], f_res_1["f"])


plt.savefig("img/f(g).png")
plt.show()

f_res_1.to_csv(f"data/free_energy_{d}.csv")
