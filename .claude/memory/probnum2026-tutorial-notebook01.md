---
name: probnum2026-tutorial-notebook01
description: Content, thesis and honest caveats of the ProbNum 2026 tutorial — notebook 01, Gaussian belief propagation as a linear solver
metadata:
  type: project
---

Tutorial for the 2nd International Conference on Probabilistic Numerics. **The talk is
`python/01-linear-systems-by-message-passing.py` alone** (decided 2026-08-24);
`02-domain-decomposition-as-message-passing.py` is kept in the repo as a backup and is not presented.
`OUTLINE.md` covers notebook 01 only. Working title: *Probabilistic numerics at scale by distributed
inference*.

**Thesis (sharpened).** Classical iterative solvers are message passing with the uncertainty deleted —
and when you put the uncertainty back, it describes the *operator*, not the *computation*. Closing that
gap is what probabilistic numerics has to offer a parallel solver.

**Arc.** §1 problem specification, opened by a story: a tiled accelerator whose control loop needs the
steady-state temperature of every tile: heat to coolant + heat to neighbours = power dissipated *is* the
five-point stencil for $(c-\Delta)u=f$, so $A$ is the floorplan, $b_i$ was measured by tile $i$ and never
assembled, and a barrier across a few thousand tiles costs more than the arithmetic. §2 classical solvers.
§3 the dictionary, $p(x)\propto e^{-\frac12 x^\top Ax + b^\top x} = \mathcal{N}(A^{-1}b, A^{-1})$, with
$(A^{-1})_{ij}$ read as the chip's thermal impedance. §4 GaBP in ~30 lines. §5 outlook. The story recurs
in §3, §4.5 and §5 by design — do not remove one callback without the others.

**Three moments meant to land:** (1) the dictionary; (2) the $k$-round belief is the exact posterior of
the $k$-hop computation tree — *and* the ceiling: the precision recursion contains no $b$, so it is not an
error estimate; (3) Jacobi = GaBP with $P_{ij}:=0$ (agrees to $6.7\times10^{-16}$).

**Verified numbers (full run, 2026-08-24).** Chain of 80: means $2.1\times10^{-13}$, variances exact, 43
rounds. 24×24 lattice, $c=0.4$: means $1.2\times10^{-12}$ in 136 rounds, BP/true variance ratio
0.909–0.987. Rounds to $10^{-10}$: CG 41, GaBP serial 67, Gauss–Seidel 110, GaBP flooding 112, Jacobi
>120. Residual after 40 rounds: Jacobi $9.0\times10^{-3}$ vs GaBP $1.9\times10^{-4}$. Rounds to $10^{-8}$,
$n:64\to4096$ — $c=2$: 22→30, $c=0.4$: 55→104, $c=0$: 124→6810. Frustrated $12\times12$ lattice:
diagonal dominance lost at $w=0.25$, walk-summability at $w\approx0.26$, still converging at $w=0.29$ (297
rounds), diverges at $w=0.30$ where $\lambda_{\min}=+0.007$ — i.e. **both sufficient conditions fail long
before anything goes wrong, and the real edge is positive definiteness**. (An earlier draft claimed
dominance was "lost almost immediately"; it is not — $4w<1$ gives exactly $w<0.25$.)

**The question the talk must survive**, now that notebook 2 is not there to answer it: *"is there any
actual advantage over the standard probnum solver?"* Three-part answer, rehearsed, part 1 volunteered
first — (1) not on iterations, not on variances, and selected inversion beats it if you can centralise;
(2) on cost structure, $O(\mathrm{nnz})$ local belief vs a dense covariance and two all-reduces per step;
(3) on what it reveals — a well-posed probnum problem inside a method thousands of people run. Evidence
for part 2 is in [[gabp-tangible-benefit-literature]].

**Caveat about the repo's own docs:** `README.md` still describes a two-notebook tutorial and frames
notebook 2 as the payoff. It contradicts `OUTLINE.md` as of 2026-08-24 and has not been updated.
