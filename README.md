# ProbNum2026-Tutorial

Tutorial for the [International Conference on Probabilistic Numerics 2026](https://probnum2026.github.io), built around message-passing Bayesian inference with [RxInfer](https://rxinfer.com).

The series covers the five classic problem settings of [probabilistic-numerics.org](https://www.probabilistic-numerics.org), each as a pair of notebooks — a Julia [Pluto](https://plutojl.org) edition in `julia/` and a Python [Marimo](https://marimo.io) edition in `python/`:

| # | Topic | Julia (Pluto) | Python (Marimo) |
|---|-------|---------------|-----------------|
| 1 | Probabilistic linear algebra | [`julia/01-probabilistic-linear-algebra.jl`](julia/01-probabilistic-linear-algebra.jl) | [`python/01-probabilistic-linear-algebra.py`](python/01-probabilistic-linear-algebra.py) |
| 2 | Bayesian quadrature | [`julia/02-bayesian-quadrature.jl`](julia/02-bayesian-quadrature.jl) | [`python/02-bayesian-quadrature.py`](python/02-bayesian-quadrature.py) |
| 3 | Probabilistic optimization | [`julia/03-probabilistic-optimization.jl`](julia/03-probabilistic-optimization.jl) | [`python/03-probabilistic-optimization.py`](python/03-probabilistic-optimization.py) |
| 4 | Probabilistic ODE solvers | [`julia/04-probabilistic-ode-solvers.jl`](julia/04-probabilistic-ode-solvers.jl) | [`python/04-probabilistic-ode-solvers.py`](python/04-probabilistic-ode-solvers.py) |
| 5 | Probabilistic PDE solvers | [`julia/05-probabilistic-pde-solvers.jl`](julia/05-probabilistic-pde-solvers.jl) | [`python/05-probabilistic-pde-solvers.py`](python/05-probabilistic-pde-solvers.py) |

## Running the Pluto notebooks

```julia
julia> import Pkg; Pkg.add("Pluto")
julia> import Pluto; Pluto.run()
```

then open the notebook file from the Pluto welcome screen. Pluto's built-in package manager installs each notebook's dependencies (RxInfer, Plots, PlutoUI, SpecialFunctions) automatically on first open — the first run takes a few minutes.

## Running the Marimo notebooks

Each `python/` notebook declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), so with [`uv`](https://docs.astral.sh/uv/) installed a notebook is self-contained:

```bash
uvx marimo edit --sandbox python/01-probabilistic-linear-algebra.py
```

Or, in an environment with `marimo`, `numpy`, `scipy` and `plotly`:

```bash
marimo edit python/01-probabilistic-linear-algebra.py
```

The Julia editions use [RxInfer](https://rxinfer.com) for the message-passing section; since there is no Python equivalent, the Marimo editions reproduce that step with a small hand-rolled Gaussian belief-propagation routine (`gaussian_bp`) and check it digit-for-digit against the closed form, exactly as the Julia notebooks check against RxInfer.
