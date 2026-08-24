---
name: probnum2026-tutorial-notebook01
description: Content, thesis and honest caveats of the ProbNum 2026 tutorial draft on Gaussian belief propagation as a linear solver
metadata:
  type: project
---

Draft tutorial for the 2nd International Conference on Probabilistic Numerics (as of 2026-08-13):
`python/01-linear-systems-by-message-passing.py` (marimo, ~1030 lines), with talk notes in
`OUTLINE.md`. Working title *Probabilistic numerics at scale by distributed inference*.

**Thesis.** Probabilistic numerics does not scale because its beliefs are *global* (dense covariance,
global policies, all-reduces per Krylov step); message passing makes them *local*, and the classical
iterative solvers turn out to be message passing with the uncertainty deleted.

**Arc.** §1 sparse $Ax=b$ as a graph (chain = tree, 2-D five-point lattice = loopy, screening $c$ sets
correlation length) → §2 classical solvers (direct/fill-in, Jacobi/GS, CG and its two barriers per
step) → §3 the dictionary: $p(x)\propto e^{-\frac12 x^\top Ax + b^\top x} = \mathcal{N}(A^{-1}b, A^{-1})$,
so matrix ↔ precision, sparsity ↔ conditional independence, solution ↔ marginal means,
$\mathrm{diag}(A^{-1})$ ↔ marginal variances, $1/A_{ii}$ ↔ *conditional* variance → §4 GaBP in ~30 lines
(two scalars per directed edge) → §5 outlook.

**Three moments meant to land:** (1) the dictionary; (2) the $k$-round belief is the exact posterior of
the $k$-hop computation tree — local, anytime uncertainty about the computation actually performed;
(3) Jacobi = GaBP with $P_{ij}:=0$ and no reverse-message exclusion (verified to $6.7\times10^{-16}$).

**Key numbers in the notebook.** Chain of 80: means/variances exact, 43 rounds. 24×24 lattice, $c=0.4$:
means exact to $10^{-12}$, BP/true variance ratio 0.909–0.987. Serial vs flooding: 67 vs 112 rounds.
Rounds to $10^{-8}$ as $n: 64\to4096$ — $c=2$: 22→30, $c=0.4$: 55→104, $c=0$: 124→6810.

**Honest caveats stated in the draft.** Loopy variances are over-confident; GaBP is *not* faster than CG
in iterations (its case is locality, asynchrony, no reductions); round count is $n$-independent only for
finite correlation length; convergence basin ends roughly where $A$ stops being positive definite and
damping does not rescue the indefinite case.

**Known gap (2026-08-13, raised by Wouter).** The notebook as it stands gives *no tangible benefit* for
the message-passing version over the standard probabilistic-numerics approach — the comparison plot has
CG winning on iterations and the BP variances being wrong. Candidate evidence for a stronger case is in
[[gabp-tangible-benefit-literature]].

Open TODOs in `OUTLINE.md`: pick a final title; possibly add a second notebook (non-symmetric /
least-squares, or a genuinely distributed application); optional Julia/RxInfer edition; slide-only intro.
