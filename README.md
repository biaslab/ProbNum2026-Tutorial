# ProbNum2026-Tutorial

Tutorial for the [International Conference on Probabilistic Numerics 2026](https://probnum2026.github.io), built around message-passing Bayesian inference with [RxInfer](https://rxinfer.com).

The series covers the five classic problem settings of [probabilistic-numerics.org](https://www.probabilistic-numerics.org), each as a pair of notebooks — a Julia [Pluto](https://plutojl.org) edition in `julia/` and a Python [Marimo](https://marimo.io) edition in `python/`:

| # | Topic | Julia (Pluto) | Python (Marimo) |
|---|-------|---------------|-----------------|
| 1 | Probabilistic linear algebra | [`julia/01-probabilistic-linear-algebra.jl`](julia/01-probabilistic-linear-algebra.jl) | planned |
| 2 | Bayesian quadrature | [`julia/02-bayesian-quadrature.jl`](julia/02-bayesian-quadrature.jl) | planned |
| 3 | Probabilistic optimization | [`julia/03-probabilistic-optimization.jl`](julia/03-probabilistic-optimization.jl) | planned |
| 4 | Probabilistic ODE solvers | [`julia/04-probabilistic-ode-solvers.jl`](julia/04-probabilistic-ode-solvers.jl) | planned |
| 5 | Probabilistic PDE solvers | [`julia/05-probabilistic-pde-solvers.jl`](julia/05-probabilistic-pde-solvers.jl) | planned |

## Running the Pluto notebooks

```julia
julia> import Pkg; Pkg.add("Pluto")
julia> import Pluto; Pluto.run()
```

then open the notebook file from the Pluto welcome screen. Pluto's built-in package manager installs each notebook's dependencies (RxInfer, Plots, PlutoUI, SpecialFunctions) automatically on first open — the first run takes a few minutes.
