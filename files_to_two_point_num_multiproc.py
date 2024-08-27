import numpy as np
import os
import re
from core.utils import get_corr_func_mom
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


d = 3
M = 32

DATA_DIRECTORY = Path("./data_enhanced/")

def compute_corr_func_from_arr(params):
    print(f"Computing correlation function im momenta picture for {params}...")
    filepath, G, gamma = params
    momenta_grid = 2 / M * np.array([[p] + [0.] * (d - 1) for p in range(M + 1)]) * np.pi
    results_list = []
    print(f"Processing file {filepath}...")

    cfgs = np.load(filepath)[np.arange(0, 10000, 10)]

    corr_f_mom = get_corr_func_mom(cfgs, momenta_grid)

    for i in range(M + 1):
        results_list.append({"g^4": G, "gamma": gamma, "D(p)": corr_f_mom.T[0, i],
                             "error": corr_f_mom.T[1, i], "p": momenta_grid.T[0, i]})

    return results_list
