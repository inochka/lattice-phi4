# Lattice Scalar Field Theory Simulation
This repository contains code for the computation of free energy and two-point functions in Euclidean Scalar Field Theory 
on a cubic dd-dimensional lattice with specified self-interaction power. The calculations use the [Hamiltonian Monte Carlo (HMC)
algorithm](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo), and errors are estimated through either jackknife or blocked resampling. Additionally,
it includes comparisons with Feynman diagrammatic techniques for weak (link) and strong (link) coupling regimes. This code
accompanies the [article](https://arxiv.org/abs/2207.11503) and serves as a supporting material for reproducing the results and analyses 
discussed therein.

This repository is based on the code from this [source](https://github.com/julian-urban).


# Initialization

To create and activate a virtual environment one should use:

```bash
python3 -m venv .venv
```

and on Linux:

```bash
source .venv/bin/activate
```

as well as on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

The code has been tested with Python 3.10 and Python 3.14. Installation (preferably in the virtual environment):

```bash
pip install -r requirements.txt
```


# Repository structure

The repository is structured as follows:

1. `core/..` - a core of numerical simulation methods. Used for import a main simulatiuon class `Lattice` and functions to compute correlators.
2. `hmc_multiprocessing.py` - a script for HMC simulations for free energy per cite (calculates field configurations and saves them as .npy files)
3. `hmc_multiprocessing_immediate_calculation.py` - a script for HMC simulations for two-point function calculation (computes the two-point estimator on the fly instead of storing complete field configurations)
4. `files_to_free_energy_num.py` - a script for calculation of free energy with error from the results of simulations (as described in numerical simulations appendix in [article](https://arxiv.org/abs/2207.11503)
5. `free_energy_error_plot.py` - a script for creating plots of free energy systematic (due to numerical integration) and random (due to HMC simulation) errors
6. `analytical_expressions.py` - a file where all analytical_expressions for comparison plots of free energy per site and two-point function are placed. This file is imported from other scripts.
7. `free_energy_comparison.py` - a script for plotting both graph of numerically simulated and analytically computed curves for free energy per site.
8. `two_point_function_comparison.py` - a script for plotting both graph of numerically simulated and analytically computed curves for two-point functions in momentum space (inside a Brillouin zone).

In the following sections we provide the description of total simulation pipelines from scratch for free energy per site and two-point function. In result one should obtain all the results described in [paper](https://arxiv.org/abs/2207.11503) including plots.

# Theoretical introduction

This section is not neccessary for the repository usage, but is useful for common understanding of a code. Here we use the traditional notations of Quantum (Statistical) Field Theory and Lattice Field Theory (for instance, [textbook](https://saalburg.aei.mpg.de/wp-content/uploads/sites/25/2017/03/wiese.pdf)). For more details one can explore the original [paper](https://arxiv.org/abs/2207.11503).

## Statistical field theory description

Let us begin with the scalar theory on a cubic Euclidean $d$-dimensional lattice with periodical boundary conditions and $M$ sites in every dimension with a power-law potential and the action of the following form:

$$
    S[\phi]=\frac{1}{2}\sum_{x, x'}L_{x,x'}\phi(x)\phi(x')+\sum_{x}V(\phi(x)),\qquad V(\phi)=\frac{1}{(2n)!}g^{2n}\phi^{2n},
$$
for some integer $n > 1$. Here all coordinate sums go over all lattice sites, which total number we denote by $N=M^d$. In the equation above $L_{x,x'}$ are the matrix elements of the operator:
$$
    L = -\alpha \triangle + \gamma
$$
 where $\alpha$ and $\gamma$ are some constants greater than zero, and: 
$$
    (\triangle f)(\vec{r})=\sum_{j=1}^{d}(f(\vec{r}+\vec{e}_{j})-2f(\vec{r})+f(\vec{r}-\vec{e}_{j})),
$$
is the lattice Laplacian. Vector $\vec{e}_{j}$ is the $j$-th element of an orthonormal frame, corresponding to the cubic lattice with unit lattice spacing in each direction. This formula is a finite difference approximation of continuous Laplacian.

For periodic boundary conditions on a cubic lattice with $M$ sites in each dimension, the eigenvalues $\{\lambda_{k} \}$ and eigenvectors $\{ h_k (x)\}$ of the operator $L$ have the form:

$$
\lambda_{k} = \gamma + 4\alpha \sum_{j=1}^{d} \sin^{2}\left(\frac{p_{j}}{2}\right),\qquad\vec{p}_{k}= \frac{2\pi\vec{k}}{M},\qquad\vec{k}\in \{0,\ldots,M-1\}^{d},\qquad h_{k}(x)=\frac{1}{\sqrt{N}}e^{i\left< p_{k},x \right>}.
$$

Here $p_k$ are conventionally called admissible lattice momenta by analogy with plane waves. 

With this action, the partition function is defined in a traditional way:

$$
    \mathcal{Z}=\int_{\mathbb{R}^{N}}\frac{\prod_{k}d\phi_k}{\sqrt{(2\pi)^ N (\det L)^{-1}}}\ \exp\left[-S[\phi] \right],
$$

For the given partition function, we understand the correlation functions or, equivalently, correlators as:

$$
    \left \langle \phi(x_1) \cdot \ldots \cdot \phi(x_k) \right \rangle = \frac{1}{\mathcal{Z}[0]} \int_{\mathbb{R}^{N}}\frac{\prod_{k}d\phi_k}{\sqrt{(2\pi)^ N \det G}} \ \phi(x_1) \cdot \ldots \cdot \phi(x_k) \ \exp \left(-S[\phi]\right),
$$

Together with the partition function, we define Free Energy Functional $\mathcal{F}$ as:

$$
    \mathcal{Z} = \exp\left(-\mathcal{F}\right).
$$

Free energy per site (in thermodynamic limit) is defined as:

$$
    f = \lim\limits_{N\rightarrow\infty} \frac{\mathcal{F}}{N},
$$
where $\mathcal{F}$ is a total free energy of a lattice system on with $N$ sites.


## Discrete Fourier Transform notion

We use the following notation for the Discrete Fourier Transform (DFT) throughout this paper. Given the lattice function $H(x)$, we will call its DFT the momentum-space function $H(p)$ satisfying the relation:

$$
    H(x) = \frac{1}{(2\pi)^d N} \sum\limits_{p} H(p) e^{-ipx}.
$$

Here the summation goes over all "admissible" momentum values, which parametrize the spectrum of lattice Laplacian from the previous section. Inverse formula is straightforward:

$$
    H(p) = \sum\limits_{x} H(x) e^{ipx}.
$$
 
Now, for $N\gg 1$ the $H(p)$ remains finite and non-zero, as well as the sum above becomes the integral:

$$
    H(x) = \frac{1}{(2\pi)^d} \int\limits_{[0;2\pi]^d} H(p) e^{-ipx} d^d p.
$$

This relation will be helpful for the study of the correlation functions in momentum representation.

## Momentum space two-point correlation function

After introducing all the neccessary notions, let us explicitly write down the expression for the two-point correlation function in the momentum representation:

$$
    \langle \phi(p) \phi(-p) \rangle = \sum\limits_x \langle \phi(0) \phi(x) \rangle e^{ipx},
$$

where we have used the translational invariance of the theory. This quantity is exactly what we will mean by the two-point fucntion in momentum space. For a free theory ($V(\phi)=0$) it has a simple form in terms of elementary functions (unlike the two-point function in coordinate space). It is also a valuable characteristic of a quantum (or statistical) field system.

To shothand the notation, we will also denote (here as well as in the code):

$$
D(p) =  \left< \phi(p) \phi(-p) \right>
$$

## Numerical simulations notes

The core of simulation code `core/..` remained almost the same as in the original [repository](github.com/julian-urban/lattice-phi4). Here one can also find links to the theoretical and numerical justification of the used HMC approach.

In the following we give some comments about the simulations results manimulations performed.

### Free eenrgy per cite $f$

Standard Monte Carlo methods do not allow direct computation of the partition function; however, they are efficient for calculating correlation functions. This is not a significant obstacle, as one can observe that:

$$
   \frac{\partial f}{\partial (g^4)} = \frac{1}{4!} \left< \phi^4 (x)\right> 
$$

for any fixed site $x$ (due to the translational invariance of the lattice under consideration with periodic boundary conditions). So, we can deduce that:

$$
    f(g) = \frac{1}{4!} \int\limits_0^g  \left< \phi^4 (x)\right>(g) d(g^4),
$$

using $f(g=0) = 0$, where correlators in integrand are considered as functions of coupling constant $g^4$. This formula used in `files_to_free_energy_num.py` script to calculate a free energy per cite.

Furthermore, the expectation value of an operator can be estimated from the distribution of numerically generated field configurations $\Phi$ as:

$$
    \left< \phi^4 (x)\right> \approx \frac{1}{|\Phi|}\sum\limits_{\phi \in \Phi} \frac{1}{N} \sum\limits_{x\in \mathbb{V}} \phi(x)^4,
$$

where we also used the averaging along the lattice sites to increase the precision.

### Two-point function in momentum space

Similarly, for the two-point correlation function, one can write:

$$
    \left< \phi (x_1) \phi(x_2)\right> \approx \frac{1}{|\Phi|}\sum\limits_{\phi \in \Phi} \phi (x_1)\phi(x_2).
$$

One can improve the precision of computations, using that the theory is translationally invariant, meaning that:

$$
    \left< \phi (x_1) \phi(x_2)\right> = \left< \phi (x_1 + \delta) \phi(x_2 + \delta) \right>,
$$

for all possible lattice shifts $\delta$. We will also use this observation in our numerical simulations, taking the average among all shifts to increase the precision of the calculations. This approach is used in `hmc_multiprocessing_immediate_calculation.py`.

To obtain the correlators in momentum space $\langle \phi(p)\phi(-p) \rangle $ as a function of admissible $p$, one can take the Discrete Fourier Transform for a grid of $p$. This step is performed in `two_point_function_comparison.py`.

## Analytical formulas

In `analytical_expressions.py` one can find analytical formulas for the perturbative expansions for free energy per site and two-point function. We do not want ro delve into the details of them here and refer to the original [paper](https://arxiv.org/abs/2207.11503).


# Simulation and comparison pipelines

In this section we describe the steps one should perform to use the presented code.

## Free energy per site

To run the numerical simulations and reproduce the results of the original [paper](https://arxiv.org/abs/2207.11503) for free energy per site $f$ one should sequentiially run the following scripts:

1. `hmc_multiprocessing.py` — in result there appears in directory `DATA_DIRECTORY` (`data` or `data_enhanced` currently) the number of files `.npy`, containing results of simulations for chosen action parameters (see the script for details)

2. `files_to_free_energy_num.py` — the script will reads all the files from `DATA_DIRECTORY`, performs calculation of $\left< \phi(x)^{4} \right>$ for chosen power interaction $\phi^{4}$, performs the numerical integration as described in theoretical introduction. Finally, it saves the results for free energy per site $f$ and its random and systematic errors in file `free_energy_{d}.csv` in folder `DATA_DIRECTORY`.

3. `free_energy_error_plot.py` — the scripts reads file `free_energy_{d}.csv` in folder `DATA_DIRECTORY` and creates plots for random and systematic error for free energy per site $f$ as a function of coupling constant $g^{4}$. The results are saved in folder `img`.

4. `free_energy_comparison.py` — the scripts reads file `free_energy_{d}.csv` in folder `DATA_DIRECTORY` and creates plots for free energy per site $f$ as a function of coupling constant $g^{4}$. In the same plot are placed analytical expressions for weak and strong coupling expansions for $f$, which can be found in `analytical_expressions.py`.

In result, in folder `img/` there will appear all the described plots for $f$.


## Two-point function in momentum space


To run the numerical simulations and reproduce the results of the original [paper](https://arxiv.org/abs/2207.11503) for two-point function in momentum space $D(p) = \left< \phi(p) \phi(-p) \right>$ one should sequentiially run the following scripts:

1. `hmc_multiprocessing_immediate_calculation.py` — in result there appears in directory `DATA_DIRECTORY` (`data` or `data_enhanced` currently) a file `two_point_data_immediate_{d}.csv` with two-point function $D(p)$ together with its error estimated with the jackknife method.

2. `two_point_function_comparison.py` — the scripts reads file `two_point_data_immediate_{d}.csv` in folder `DATA_DIRECTORY` and creates plots for two-point function $D(p)$ as a function of momentum $p$ for different $g^{4}$. In the same plot are placed analytical expressions for weak and strong coupling expansions for $D(p)$, which can be found in `analytical_expressions.py`. To avoid cluttering the plot, the comparison with strong- and weak-coupling expansions is presented in separate plots. For details one can check the script itself. To choose the mode of plot, one should specify the variable `PLOT_REGIME` to one of "strong" or "weak" values.

In result, in folder `img/` there will appear all the described plots for $D(p)$.




