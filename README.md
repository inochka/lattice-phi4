# Lattice Scalar Field Theory Simulation

This repository contains the numerical simulations used in the article
[Strong-Weak Coupling Duality in Scalar Lattice QFT with Application to Strong Coupling Decompositions](https://arxiv.org/abs/2207.11503).
It computes the free energy per lattice site and the momentum-space two-point function for Euclidean scalar field theory on a periodic, $d$-dimensional hypercubic lattice.

The numerical calculations use Hamiltonian Monte Carlo (HMC). The two simulation pipelines are intentionally independent:

- **Free energy:** HMC evaluation of $\langle \phi^4 \rangle$, followed by numerical integration over $g^4$.
- **Two-point function:** HMC evaluation of $D(p)=\langle\phi(p)\phi(-p)\rangle$ on the admissible lattice-momentum grid.

The numerical results can be compared with the weak- and strong-coupling expressions implemented in `analytical_expressions.py`.

The HMC implementation in `core/` is derived from
[`julian-urban/lattice-phi4`](https://github.com/julian-urban/lattice-phi4).

## Quick start

The code has been tested with Python 3.10 and Python 3.14.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python3 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the runtime dependencies with `pip`:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Run the automated checks:

```bash
python3 -m unittest discover -s tests
```

### Small smoke runs

The smoke configurations use small lattices and short Markov chains. They verify that both pipelines execute correctly; they are not intended to reproduce the article's numerical precision.

```bash
python3 run_pipeline.py free-energy --config configs/free_energy_smoke.json --overwrite
```

```bash
python3 run_pipeline.py two-point --config configs/two_point_smoke.json --overwrite
```

All paths in the configuration files are resolved relative to the repository root, so the commands may be launched from another working directory as well.

## Reproducing the simulation pipelines

### Free energy

Run the complete numerical pipeline:

```bash
python run_pipeline.py free-energy --config configs/free_energy.json --overwrite
```

This command performs two steps:

1. `hmc_multiprocessing.py` runs the HMC simulations and writes the measured $\langle\phi^4\rangle$ values.
2. `files_to_free_energy_num.py` interpolates these values and performs the numerical integration over $g^4$.

To generate the comparison and error plots as well:

```bash
python run_pipeline.py free-energy --config configs/free_energy.json --with-plots
```

The analytical curves are cached under `cache/theory/`. Use `--recompute-theory` to ignore a compatible cache.


### Two-point function

Run the HMC calculation:

```bash
python run_pipeline.py two-point --config configs/two_point.json --overwrite
```

Generate a strong-coupling comparison plot:

```bash
python run_pipeline.py two-point --config configs/two_point.json --skip-simulation --with-plots --regime strong
```

For the weak-coupling comparison, replace `strong` with `weak`.

The two-point pipeline retains the required field configurations in memory while the correlation estimator is evaluated, but does not save them as `.npy` files.

## Resume and overwrite behavior

Both simulation scripts support resumable execution.

- Without `--overwrite`, parameter combinations already present in the configured output CSV are skipped.
- With `--overwrite`, the existing output CSV is discarded and all requested combinations are recomputed.
- Results are written atomically after every completed coupling, reducing the risk of a corrupted CSV after an interrupted run.

Each output row contains the relevant lattice parameters, HMC settings (including the base leapfrog-step count), acceptance rate, retained-sample count, and random seed.

When `base_seed` is `null`, each task receives an independent seed from operating-system entropy. Set `base_seed` to an integer in the configuration file to obtain deterministic task seeds. Independent per-task seeding also prevents separate multiprocessing workers from inheriting identical NumPy random states.

## Configuration

The production and smoke-test configurations are stored in `configs/`:

```text
configs/
├── free_energy.json
├── free_energy_smoke.json
├── two_point.json
└── two_point_smoke.json
```

The production configurations contain the parameters used for the full numerical calculations. The smoke-test configurations use substantially smaller lattices and shorter Markov chains and are intended only to verify that the installation and complete pipeline work correctly.

All relative paths are resolved from the repository root.

### Lattice parameters

```json
"lattice": {
  "size": 8,
  "dimension": 2,
  "alpha": 1.0,
  "gammas": [1.0]
}
```

* `size`: number of lattice sites ($M$) along each spatial direction. The total number of sites is $N=M^d$.
* `dimension`: lattice dimension ($d$).
* `alpha`: coefficient of the lattice Laplacian in the quadratic part of the action:
  $$
  L=-\alpha\Delta+\gamma.
  $$
* `gammas`: values of the quadratic coefficient (\gamma). The pipeline performs an independent calculation for every value in this list.

### Coupling constants

```json
"couplings_g4": [
  0.0,
  0.625,
  1.25,
  1.875,
  2.5
]
```

`couplings_g4` specifies the values of the quartic coupling ($g^4$) at which HMC simulations are performed.

For the free-energy calculation, these values also form the numerical integration grid. A denser grid generally reduces the interpolation and integration error, while increasing the total simulation time.

The values should be given in ascending order and should include $(g^4=0)$ when the free energy is obtained by integration from the free theory.

### HMC parameters

```json
"hmc": {
  "warmup_steps": 1000,
  "production_steps": 10000,
  "sample_every": 10,
  "processes": 6,
  "base_seed": null,
  "base_leapfrog_steps": 100
}
```

* `warmup_steps`: number of initial HMC trajectories discarded before measurements are collected. These trajectories allow the Markov chain to approach the target distribution.
* `production_steps`: number of HMC trajectories generated after the warm-up stage.
* `sample_every`: measurement interval in HMC trajectories. For example, a value of `10` means that an observable is recorded after every tenth production trajectory.
* `processes`: maximum number of worker processes used to run independent parameter combinations in parallel. Increasing this value can reduce wall-clock time but also increases CPU and memory usage.
* `base_seed`: base random seed used to derive independent seeds for individual simulations.

  * An integer gives reproducible random-number streams.
  * `null` requests non-deterministic seeds from the operating system.
* `base_leapfrog_steps`: baseline number of leapfrog integration steps used by the HMC integrator. This parameter controls the numerical integration of the molecular-dynamics trajectory and should normally be changed only together with the corresponding HMC tuning and acceptance-rate checks.

The approximate number of stored measurements per parameter combination is

$$
N_{\mathrm{samples}} \approx
\frac{\texttt{production\_steps}}
{\texttt{sample\_every}}
$$

Warm-up trajectories are not included in this number.

### Free-energy integration

```json
"integration": {
  "interpolation": "cubic"
}
```

* `interpolation`: interpolation method applied to the simulated values of $ \left\langle \phi^4 \right\rangle$ before numerical integration over $(g^4)$.

The production configuration uses cubic interpolation. Its accuracy depends on the density and placement of the values in `couplings_g4`, particularly in regions where the observable changes rapidly.

### Paths

```json
"paths": {
  "observables": "results/generated/free_energy/phi4_d2.csv",
  "free_energy": "results/generated/free_energy/free_energy_d2.csv",
  "figures": "figures/generated/free_energy",
  "theory_cache": "cache/theory"
}
```

* `observables`: CSV file containing the observables measured directly in the HMC simulations, together with their statistical uncertainties.
* `free_energy`: CSV file containing the free energy obtained by numerical integration and the associated statistical and integration errors.
* `figures`: directory in which generated plots are stored.
* `theory_cache`: directory used to cache analytical curves whose evaluation is comparatively expensive.

Generated numerical results, cached values, and figures are kept in separate directories so that the source code and configuration files remain unchanged during a simulation run.

### Plotting parameters

```json
"plot": {
  "weak_g_min": 0.0,
  "weak_g_max": 2.5,
  "weak_points": 150,
  "strong_g_min": 0.1,
  "strong_g_max": 2.8,
  "strong_points": 100,
  "x_g4_min": 0.0,
  "x_g4_max": 50.0,
  "y_min": 0.0,
  "y_max": 0.20,
  "theory_seed": null
}
```

* `weak_g_min`, `weak_g_max`: range of (g) used to evaluate the weak-coupling analytical approximation.
* `weak_points`: number of points in the weak-coupling analytical grid.
* `strong_g_min`, `strong_g_max`: range of (g) used to evaluate the strong-coupling analytical approximation.
* `strong_points`: number of points in the strong-coupling analytical grid.
* `x_g4_min`, `x_g4_max`: displayed horizontal-axis range in terms of (g^4).
* `y_min`, `y_max`: displayed vertical-axis range.
* `theory_seed`: random seed used by analytical calculations that involve stochastic numerical evaluation.

  * An integer makes the cached analytical curves reproducible.
  * `null` uses a non-deterministic seed.

The analytical-grid ranges determine where the theoretical approximations are evaluated; they do not change the HMC simulation points. The displayed axis ranges affect only the generated figures.

The production configuration values reproduce the constants that were previously embedded directly in the simulation and plotting scripts. Changes to a configuration file therefore provide a complete record of the parameters used for a particular numerical experiment.


## Output layout

Generated files are kept separate from the committed reference data:

```text
results/
├── generated/       # new simulation results; ignored by Git
└── reference/       # selected results supplied with the repository

figures/
└── generated/       # generated figures; ignored by Git

cache/
└── theory/          # cached analytical curves; ignored by Git
```

The selected reference CSV files can be plotted directly, for example:

```bash
python free_energy_comparison.py --config configs/free_energy.json --data results/reference/free_energy/free_energy_d3.csv
```

```bash
python two_point_function_comparison.py --config configs/two_point.json --data results/reference/two_point/two_point_d3_averaged.csv --regime strong
```

## Repository structure

```text
.
├── analytical_expressions.py
├── core/
├── configs/
├── results/reference/
├── tests/
├── simulation_utils.py
├── run_pipeline.py
├── hmc_multiprocessing.py
├── files_to_free_energy_num.py
├── free_energy_comparison.py
├── free_energy_error_plot.py
├── hmc_multiprocessing_immediate_calculation.py
└── two_point_function_comparison.py
```

- `core/` contains the HMC lattice implementation and correlation utilities.
- `analytical_expressions.py` contains the checked analytical weak- and strong-coupling expressions.
- `simulation_utils.py` contains path, configuration, random-seed, cache, and atomic-I/O helpers.
- `run_pipeline.py` is the user-facing entry point for the two independent pipelines.

## Theory and conventions

### Lattice action

Consider a scalar field on a periodic, $d$-dimensional hypercubic lattice with $M$ sites in every dimension and $N=M^d$ total sites. The action is

$$
S[\phi]
=
\frac{1}{2}\sum_{x,x'}L_{x,x'}\phi(x)\phi(x')
+
\sum_x V(\phi(x)),
\qquad
V(\phi)=\frac{g^{2n}}{(2n)!}\phi^{2n},
$$

where $n>1$ and

$$
L=-\alpha\triangle+\gamma,
$$

with $\alpha>0$ and $\gamma>0$. The lattice Laplacian is

$$
(\triangle f)(\vec r)
=
\sum_{j=1}^{d}
\left[
 f(\vec r+\vec e_j)-2f(\vec r)+f(\vec r-\vec e_j)
\right].
$$

For periodic boundary conditions, the eigenvalues and eigenvectors of $L$ are

$$
\lambda_k
=
\gamma+4\alpha\sum_{j=1}^{d}\sin^2\left(\frac{p_j}{2}\right),
\qquad
\vec p_k=\frac{2\pi\vec k}{M},
\qquad
\vec k\in\{0,\ldots,M-1\}^{d},
$$

$$
h_k(x)=\frac{1}{\sqrt N}e^{i\langle p_k,x\rangle}.
$$

The simulations in this repository are performed at finite $M$. A sufficiently large lattice is used as an approximation to the large-volume regime; no finite-size extrapolation is performed by these scripts.

### Partition function and free energy

The normalized partition function is

$$
\mathcal Z
=
\int_{\mathbb R^N}
\frac{\prod_x d\phi_x}
{\sqrt{(2\pi)^N(\det L)^{-1}}}
\exp[-S[\phi]],
$$

and the free energy is defined by

$$
\mathcal Z=e^{-\mathcal F}.
$$

The numerical quantity reported by the scripts is the finite-lattice free energy per site,

$$
f_M=\frac{\mathcal F}{N}.
$$

### Discrete Fourier transform

The convention used in the article and in this repository is

$$
H(x)=\frac{1}{(2\pi)^dN}\sum_p H(p)e^{-ipx},
$$

with inverse transform

$$
H(p)=\sum_x H(x)e^{ipx}.
$$

The sum is taken over admissible lattice momenta. For a large lattice, it approaches

$$
H(x)=\frac{1}{(2\pi)^d}\int_{[0,2\pi]^d}H(p)e^{-ipx}\,d^dp.
$$

### Momentum-space two-point function

Using translational invariance,

$$
D(p)
=
\langle\phi(p)\phi(-p)\rangle
=
\sum_x\langle\phi(0)\phi(x)\rangle e^{ipx}.
$$

The estimator averages over all lattice translations before transforming to momentum space. The implementation is contained in `core/utils.py` and is called by `hmc_multiprocessing_immediate_calculation.py`.

### Free-energy integration

For the quartic interaction used by the numerical free-energy pipeline,

$$
\frac{\partial f_M}{\partial(g^4)}
=
\frac{1}{4!}\langle\phi^4(x)\rangle.
$$

Therefore,

$$
f_M(g)-f_M(0)
=
\frac{1}{4!}\int_0^{g^4}
\langle\phi^4(x)\rangle_{G}\,dG.
$$

The current implementation uses cubic interpolation between the simulated coupling values and adaptive quadrature for the final integral. The column `quadrature_error` contains the error estimate returned by the numerical quadrature routine. The column `phi4_naive_standard_error` is retained as a diagnostic for the sampled observable and is not propagated into `f_M` by the current pipeline; this preserves the calculation used for the article.

## Notes on computational cost

The production configurations are substantially more expensive than the smoke configurations. Analytical strong-coupling expressions also contain multidimensional numerical integrations. Their results are cached using a hash of all parameters that affect the curve, preventing stale cache reuse after a configuration change.

## Citation

Citation metadata is provided in `CITATION.cff`. The accompanying article is:

> Nikita A. Ignatyuk and Daniel Skliannyi, *Strong-Weak Coupling Duality in Scalar Lattice QFT with Application to Strong Coupling Decompositions*, arXiv:2207.11503.
