# ProbNum2026-Tutorial

Tutorial for the [2nd International Conference on Probabilistic Numerics](https://probnum2026.github.io).

**Solving systems of equations with distributed probabilistic numerics** — probabilistic numerical
methods have a scaling problem: their beliefs are global objects with dense covariances, and their
iterations need global synchronisation. This tutorial takes the opposite route. Reading the sparsity
pattern of a linear system as a graphical model turns solving $Ax=b$ into marginal inference in a
Gaussian Markov random field, and the solver into **message passing between neighbouring unknowns** —
local, asynchronous, communication-light, and with a per-node uncertainty as a by-product.

The punchline of the first notebook: the Jacobi method *is* Gaussian belief propagation with the
second moment deleted (verified to machine precision in the notebook). The classical iterative solver
is not an alternative to the probabilistic one; it is the probabilistic one, marginalised down to a
point estimate.

## Structure

| # | Notebook | Contents |
|---|----------|----------|
| 1 | [`python/01-linear-systems-by-message-passing.py`](python/01-linear-systems-by-message-passing.py) | Problem specification → classical solvers → the probabilistic-numerics view → Gaussian belief propagation |
| 2 | [`python/02-domain-decomposition-as-message-passing.py`](python/02-domain-decomposition-as-message-passing.py) | Parallel PDE solvers as block GaBP: Schur complements as messages, and what the belief does not know |

Notebook 1 exists in two editions, which compute the same things and agree to the digits shown:

| edition | file |
|---|---|
| Python / marimo | [`python/01-linear-systems-by-message-passing.py`](python/01-linear-systems-by-message-passing.py) |
| Julia / Pluto | [`julia/01-linear-systems-by-message-passing.jl`](julia/01-linear-systems-by-message-passing.jl) |

Notebook 1 makes the *structural* argument and is honest that, on one laptop with a 2-D lattice, CG wins
on iterations and the BP variances are over-confident. Notebook 2 puts the same construction where
parallel PDE solvers actually live — a domain decomposed across ranks — and turns the "so what?" into a
probabilistic-numerics question rather than an engineering one.

See [`OUTLINE.md`](OUTLINE.md) for the talk narrative, timings and rehearsal notes.

## The two arguments

**Notebook 1 — the reframing.** $p(x) \propto e^{-\frac12 x^\top A x + b^\top x} = \mathcal{N}(A^{-1}b, A^{-1})$,
so the matrix is a precision matrix, its sparsity is a conditional-independence graph, the solution is a
vector of marginal means, and $\mathrm{diag}(A^{-1})$ is the marginal variances. Belief propagation on that
graph is a solver; Jacobi is that solver with the second moment deleted.

**Notebook 2 — domain decomposition, and the gap it exposes.** Group the unknowns by subdomain and the
factor graph coarsens to one node per rank. Then:

* **Domain decomposition methods *are* message passing.** The message from subdomain $i$ to $j$ is exactly
  the **Schur complement** $-A_{ij}^\top A_{ii}^{-1} A_{ij}$ — the discrete Dirichlet-to-Neumann map,
  verified to machine zero. On a chain of subdomains, GaBP *is* block substructuring, converging in
  exactly $P-1$ rounds.
* **Granularity is a dial** from a direct solver (one subdomain, exact variances, one round) to scalar
  belief propagation (one node per subdomain, most iterations, most bias). Coarse blocks absorb their
  internal loops exactly — cluster-variation correction, reached from the numerics side.
* **The parallel cost argument** is real but modest: a GaBP round is a halo exchange, a CG iteration is a
  halo exchange plus an all-reduce barrier. Worth a constant factor, more under load imbalance.
* **The belief is blind to its own numerical error** — and this is the point. The precision recursion
  contains no $b$, so the reported uncertainty is *bit-identical* for two right-hand sides whose actual
  errors differ several-fold. Meanwhile the question a PDE solver most needs answered — *when do I stop?*
  — is decided by the discretisation error, which the belief also cannot see.

The third bullet is the probabilistic-numerics content; the first two are what make it a statement about
a method thousands of people already run.

## Running the notebooks

Each notebook declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), so with
[`uv`](https://docs.astral.sh/uv/) installed it is self-contained:

```bash
uvx marimo edit --sandbox python/01-linear-systems-by-message-passing.py
uvx marimo edit --sandbox python/02-domain-decomposition-as-message-passing.py
```

Or, in an environment with `marimo`, `numpy`, `scipy` and `plotly`:

```bash
marimo edit python/01-linear-systems-by-message-passing.py
```

The Julia edition needs [Pluto](https://plutojl.org); it manages its own package environment, so a
first open resolves `Plots` and `PlutoUI` (network required once):

```julia
julia> using Pluto; Pluto.run()
```

then open `julia/01-linear-systems-by-message-passing.jl` from the Pluto start page. All cells run in
well under a second once the packages are compiled; the first load pays the usual Plots compilation.

Notebook 1 runs end-to-end in about ten seconds. Notebook 2 is heavier on first load — it sweeps
decomposition granularities and straggler slowdowns, each of which runs the solver to convergence — so
open it before a session rather than restarting it live.

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
