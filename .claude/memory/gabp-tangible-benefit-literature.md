---
name: gabp-tangible-benefit-literature
description: Literature where Gaussian belief propagation demonstrably beats standard solvers — the evidence base for the ProbNum 2026 tutorial's "why bother" question
metadata:
  type: reference
---

Search done 2026-08-13 for settings where message passing has a *measured* advantage over classical /
standard-probnum linear solvers. Five clusters, strongest first. Supports [[probnum2026-tutorial-notebook01]].

**1. Message passing on message-passing hardware (strongest wall-clock evidence).**
Ortiz, Pupilli, Leutenegger & Davison, *Bundle Adjustment on a Graph Processor*, CVPR 2020
(arXiv:2003.03134): 125 keyframes / 1919 points in <40 ms on one 1216-core Graphcore IPU vs 1450 ms for
Ceres on CPU (~36×). Ortiz, Evans & Davison, *A visual introduction to Gaussian Belief Propagation*
(arXiv:2107.02308) is the manifesto: local storage/processing, arbitrary schedules, no global coordination.

**2. Matrix-free FEM / multigrid.** Ahmadi & Giannacopoulos (McGill): FMGaBP reformulates FEM as
distributed variational inference, *eliminating all global algebraic operations and sparse data
structures* — element-local stencil updates only. 2.9× over parallel MG-PCG on 8 cores, with
discretisation-independent convergence rate. Their tuned GaBP solver reports up to 6× fewer iterations and
1.8× faster execution than diagonally-preconditioned CG (and 17× over the earlier GaBP for FEM matrices).
Refs: *Parallel finite element technique using GaBP*, Comput. Phys. Commun. 2015; *Efficient implementation
of GaBP solver for large sparse diagonally dominant linear systems* (McGill escholarship).

**3. Where the factor graph *is* the machine — no centralisation possible.**
- Murai, Ortiz, Saeedi, Kelly & Davison, *A Robot Web for Distributed Many-Device Localisation*,
  T-RO 2023 (arXiv:2202.03314) + Science Robotics 2024: up to 1000 robots, ad-hoc peer-to-peer, accuracy
  matching a centralised nonlinear factor-graph solver, robust to asynchrony and dropped messages.
- Patwardhan, Murai & Davison, *Distributing Collaborative Multi-Robot Planning with GaBP*, RA-L 8(2) 2023:
  distributed planning that degrades gracefully under communication failure.
- Hug, Alzugaray & Chli, *Hyperion*, ECCV 2024 (arXiv:2407.07074): symbolic continuous-time GBP framework,
  decentralised inference across agents, 2.4–110× faster spline implementations.
- Ćošović & Vukobratović, power-system state estimation (arXiv:1605.08296, 1705.01376): real-time,
  distributable, and *robust to ill-conditioning from wildly differing measurement variances* — a
  qualitative advantage over matrix-based WLS, which needs observability analysis.
- Bickson et al., distributed Kalman filter / linear MMSE via GaBP (arXiv:0810.1628).
- Wireless: Gaussian message-passing detectors for massive MIMO reach MMSE performance without the
  $O(n^3)$ inverse; variances provably converge to the MMSE MSE.

**4. Asynchrony as a first-class property.** Convergence under arbitrary/asynchronous schedules with no
global clock, verified up to $10^6\times10^6$; asynchronous runs converged wherever synchronous did, and
often *faster*. See also *Distributed convergence verification for GaBP* (arXiv:1711.09888) and Su & Wu,
JMLR 20 (2019) for arbitrary-size nodes.

**5. Probnum-adjacent.** *Scalable Data Assimilation with Message Passing* (arXiv:2404.12968) — NWP-scale
assimilation as Bayesian inference, motivated explicitly by synchronisation overhead; notes that loopy
message passing gets the mean right but the marginal uncertainties are biased.

**Counterpoint to state honestly.** If all you want is $\mathrm{diag}(A^{-1})$ on a sparse SPD system,
Takahashi/selected inversion off a sparse Cholesky is exact and very fast (Rue & Martino; INLA), so GaBP
does not win on variances — it wins where the *factorisation itself* is unaffordable or impossible to
centralise, or where the hardware is a network.
