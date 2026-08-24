# ProbNum2026-Tutorial

Tutorial for the [2nd International Conference on Probabilistic Numerics](https://probnum2026.github.io).

**Solving systems of equations with distributed probabilistic numerics** — probabilistic numerical
methods have a scaling problem: their beliefs are global objects with dense covariances, and their
iterations need global synchronisation. This tutorial takes the opposite route. Reading the sparsity
pattern of a linear system as a graphical model turns solving $Ax=b$ into marginal inference in a
Gaussian Markov random field, and the solver into **message passing between neighbouring unknowns** —
local, asynchronous, communication-light, and with a per-node uncertainty as a by-product.

The punchline: the Jacobi method *is* Gaussian belief propagation with the second moment deleted
(verified to machine precision in the notebook). The classical iterative solver is not an alternative
to the probabilistic one; it is the probabilistic one, marginalised down to a point estimate.

## Structure

The session is one notebook, which exists in two editions that compute the same things and agree to
the digits shown:

| edition | file |
|---|---|
| Python / marimo | [`python/01-linear-systems-by-message-passing.py`](python/01-linear-systems-by-message-passing.py) |
| Julia / Pluto | [`julia/01-linear-systems-by-message-passing.jl`](julia/01-linear-systems-by-message-passing.jl) |

It runs Problem specification → classical solvers → the probabilistic-numerics view → Gaussian belief
propagation → research outlook, and is honest that, on one laptop with a 2-D lattice, CG wins on
iterations and the BP variances are over-confident.

The notebook covers the classical solvers in two paragraphs. Three companion notebooks in
[`extra/`](extra/) are the long version of those paragraphs, for participants who want the classical
method in full before it reappears as message passing. They are self-contained and are not presented
in the session.

| # | Notebook | Contents |
|---|----------|----------|
| 1a | [`extra/01a-jacobi.py`](extra/01a-jacobi.py) | The splitting, the iteration matrix, the exact spectrum of the stencil, damped Jacobi and the smoothing factor |
| 1b | [`extra/01b-gauss-seidel.py`](extra/01b-gauss-seidel.py) | Use-what-has-arrived, ρ_GS = ρ_J², why the ordering is a real choice, red–black parallelism, SOR and the optimal ω |
| 1c | [`extra/01c-krylov.py`](extra/01c-krylov.py) | Krylov subspaces, CG as energy-norm optimality, the Chebyshev bound and why it is loose, eigenvalue clustering, the two all-reduces, preconditioning, BayesCG |


## Installation

### Python / marimo

The notebooks declare their dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), so
the least invasive route is [`uv`](https://docs.astral.sh/uv/), which builds a throwaway environment
per notebook and installs nothing globally.

**1. Install `uv`** (skip if you have it — check with `uv --version`):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`brew install uv`, `pipx install uv` and `pip install uv` also work.

**2. Open the notebook.** Nothing else to install — `--sandbox` reads the dependency block and
provisions Python, marimo, NumPy, SciPy and Plotly on first run:

```bash
uvx marimo edit --sandbox python/01-linear-systems-by-message-passing.py
```

A browser tab opens at `http://localhost:2718`. The first launch downloads packages (tens of
seconds); later launches are instant.

<details>
<summary>Alternative: a conventional virtual environment (no <code>uv</code>)</summary>

Requires Python ≥ 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install marimo numpy scipy plotly
marimo edit python/01-linear-systems-by-message-passing.py
```

Conda works the same way:
`conda create -n probnum2026 python=3.12 numpy scipy plotly && conda activate probnum2026 && pip install marimo`.

</details>

### Julia / Pluto

**1. Install Julia** via [`juliaup`](https://github.com/JuliaLang/juliaup) (skip if `julia --version`
already reports ≥ 1.10):

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows
winget install julia -s msstore
```

**2. Install Pluto**, once, into your default Julia environment:

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'
```

**3. Start Pluto and open the notebook:**

```bash
julia -e 'using Pluto; Pluto.run()'
```

A browser tab opens at `http://localhost:1234`. Paste the path to
`julia/01-linear-systems-by-message-passing.jl` into the *Open a notebook* box on the start page.

Pluto notebooks carry their own package environment, so the first open resolves and precompiles
`Plots` and `PlutoUI` by itself — **network access is required once**, and that first load takes a
few minutes, essentially all of it Plots precompilation. Do this before the session, not during it.

### Verifying the install before the session

Both editions run end-to-end in about ten seconds once packages are in place. A good check is to open
the notebook, run all cells, and confirm the last section renders its plots — if the interactive
sliders in §4 respond, everything is wired up correctly.

## Key references

* O. Shental, D. Bickson, P. H. Siegel, J. K. Wolf & D. Dolev (2008).
  *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867.
  [doi:10.1109/ISIT.2008.4595311](https://doi.org/10.1109/ISIT.2008.4595311) ·
  extended version [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
  *The core algorithm: the GaBP message updates, the equivalence of GaBP on trees with Gaussian
  elimination, and of GaBP with the precision messages clamped to zero with Jacobi.*
* D. Bickson, Y. Tock, O. Shental & D. Dolev (2008).
  *Polynomial linear programming with Gaussian belief propagation*. Allerton, 895–901.
  [doi:10.1109/ALLERTON.2008.4797652](https://doi.org/10.1109/ALLERTON.2008.4797652).
  *Interior-point LP where each Newton step is solved by message passing.*
* V. Fanaskov (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear
  equations*. SIAM J. Sci. Comput. 44(2), A77–A102.
  [doi:10.1137/19M1275139](https://doi.org/10.1137/19M1275139) ·
  [arXiv:1904.04093](https://arxiv.org/abs/1904.04093).
  *Non-symmetric GaBP, a generalised-BP (cluster variation) matrix inversion, links to LU and
  block-LU, and GaBP as a multigrid smoother.*
* Y. Weiss & W. T. Freeman (2001). *Correctness of belief propagation in Gaussian graphical models of
  arbitrary topology*. Neural Computation 13(10), 2173–2200. *Means exact on convergence, variances not.*
* D. M. Malioutov, J. K. Johnson & A. S. Willsky (2006). *Walk-sums and belief propagation in Gaussian
  graphical models*. JMLR 7, 2031–2064. *Walk-summability, and why the BP variances are over-confident.*

## Archive

`archive/` holds the earlier, broader draft of the tutorial: five paired Julia/Pluto and Python/Marimo
notebooks covering the five classic problem settings of
[probabilistic-numerics.org](https://www.probabilistic-numerics.org). It is kept for reference and for
material that may be recycled — the current tutorial deliberately goes narrow and deep instead.
