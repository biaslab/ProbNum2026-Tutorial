# ProbNum2026-Tutorial

Tutorial for the [2nd International Conference on Probabilistic Numerics](https://probnum2026.github.io).

**Probabilistic numerics at scale by distributed inference** 
In this tutorial, we are looking at the sparsity pattern of the $A$ matrix in a system 
of linear equations as a graphical model and turn solving $Ax=b$ into marginal inference in a
Gaussian Markov random field, and the solver into a message passing procedure that is
local, asynchronous, communication-light, and with a per-node uncertainty as a by-product.

## Installation

The notebooks complement each other:

* `demo.py` (Python / marimo) develops linear solvers as distributed Gaussian inference.
* `demo.jl` (Julia / Pluto), **Keep the die cool**, uses RxInfer to solve a chip's temperature
  field. Adjust four periodic cooling amplitudes and phases, keep the die-average temperature
  at or below 20°C throughout the cycle, then reduce cooling effort. The field has no heat
  storage; all time dependence lives in the inputs. The notebook includes live plots,
  playback, a full-cycle score, and a collapsible benchmark.
* `deeper_tutorial.jl` (Julia / Pluto) reproduces RxInfer's **Solving Linear Systems with
  Message Passing** example end to end: chains and loops, a heat grid, a state-space prior
  with missing observations, and the composed model with its animation. Numerical claims
  are checked against an exact reduced Gaussian reference.

The notebooks carry their own dependency lists. The previous Julia edition of the linear-solver
tutorial is preserved in `archive/julia/linear-systems-by-message-passing.jl`.

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
environment, so the first open resolves and precompiles `RxInfer`, `ReactiveMP`, `Plots`,
`PlutoUI`, and `HypertextLiteral` by itself —
**network access is required once**, and that takes a few minutes. Do this before the session,
not during it. The notebook then computes five checked message-passing responses once;
editing cooling fields or playing the animation reuses those responses.

The cooling exercise adapts [RxInfer's linear-systems example](https://examples.rxinfer.com/categories/advanced_examples/solving_linear_systems_with_message_passing/).
Its target is the spatial average at every time, with the hottest tile displayed separately.
Cooling effort is an illustrative heat-removal measure, not calibrated electrical energy.

To run the Julia model and scoring checks, use `julia scripts/check-cooling.jl`. This provisions
the notebook's embedded dependencies in a temporary project and checks the reference schedules,
RxInfer accuracy, and continuous-cycle scoring.

Open the complete RxInfer tutorial with:

```bash
julia -e 'using Pluto; Pluto.run(notebook="deeper_tutorial.jl")'
```

This notebook includes a pinned environment, the upstream source revision and its MIT notice.
The final model has 120 time slices and 48,480 Gaussian coordinates; inference and the
132-frame animation can take several minutes on the first run. The text distinguishes
convergence error from error against simulated truth and explains the original model's
uncancelled normaliser bias.

Use Julia 1.11 or later for `deeper_tutorial.jl` (validated on 1.11.9). Run its complete
numerical and animation checks with `julia scripts/check-deeper-tutorial.jl`.

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
