# ProbNum2026-Tutorial

Tutorial for the [2nd International Conference on Probabilistic Numerics](https://probnum2026.github.io).

**Probabilistic numerics at scale by distributed inference** 

In this tutorial, we are looking at the sparsity pattern of the $A$ matrix in a system 
of linear equations as a graphical model and turn solving $Ax=b$ into marginal inference in a
Gaussian Markov random field, and the solver into a message passing procedure that is
local, asynchronous, communication-light, and with a per-node uncertainty as a by-product.

## Installation

Pick whichever language you prefer — `demo.py` (Python) and `demo.jl` (Julia) cover the same
material. Both notebooks carry their own dependency list, so there is nothing to install beyond
the language toolchain itself.

### Python / marimo

**1. Install [`uv`](https://docs.astral.sh/uv/)** (skip if `uv --version` already works):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

**2. Open the notebook.** `--sandbox` provisions Python, marimo, NumPy, SciPy and Plotly in a
throwaway environment, installing nothing globally:

```bash
uvx marimo edit --sandbox demo.py
```

A browser tab opens at `http://localhost:2718`. The first launch downloads packages (tens of
seconds); later launches are instant. The same command opens the side notebooks in `extra/`.

*Without `uv`:* in any Python ≥ 3.11 environment, `pip install marimo numpy scipy plotly` and then
`marimo edit demo.py`.

### Julia / Pluto

**1. Install Julia** via [`juliaup`](https://github.com/JuliaLang/juliaup) (skip if `julia --version`
reports ≥ 1.10):

```bash
curl -fsSL https://install.julialang.org | sh  # macOS / Linux
winget install julia -s msstore                # Windows
```

**2. Open the notebook.** This installs Pluto once, then starts it on `demo.jl`:

```bash
julia -e 'using Pkg; Pkg.add("Pluto"); using Pluto; Pluto.run(notebook="demo.jl")'
```

A browser tab opens at `http://localhost:1234`. Pluto notebooks carry their own package
environment, so the first open resolves and precompiles `Plots` and `PlutoUI` by itself —
**network access is required once**, and that takes a few minutes. Do this before the session,
not during it.

## Key references

* O. Shental, D. Bickson, P. H. Siegel, J. K. Wolf & D. Dolev (2008).
  *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867.
  [doi:10.1109/ISIT.2008.4595311](https://doi.org/10.1109/ISIT.2008.4595311) ·
  extended version [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
* D. Bickson, Y. Tock, O. Shental & D. Dolev (2008).
  *Polynomial linear programming with Gaussian belief propagation*. Allerton, 895–901.
  [doi:10.1109/ALLERTON.2008.4797652](https://doi.org/10.1109/ALLERTON.2008.4797652).
* V. Fanaskov (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear
  equations*. SIAM J. Sci. Comput. 44(2), A77–A102.
  [doi:10.1137/19M1275139](https://doi.org/10.1137/19M1275139) ·
  [arXiv:1904.04093](https://arxiv.org/abs/1904.04093).
* Y. Weiss & W. T. Freeman (2001). *Correctness of belief propagation in Gaussian graphical models of
  arbitrary topology*. Neural Computation 13(10), 2173–2200. 
* D. M. Malioutov, J. K. Johnson & A. S. Willsky (2006). *Walk-sums and belief propagation in Gaussian
  graphical models*. JMLR 7, 2031–2064. 
