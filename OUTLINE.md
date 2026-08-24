# Tutorial outline — Solving systems of equations with distributed probabilistic numerics

Working title, in order of preference:

1. **Probabilistic numerics at scale by distributed inference** — names both the payoff (scale) and the mechanism. Best if the tutorial is pitched as *"here is what probnum is missing"*.
2. *A distributed inference perspective to probabilistic numerics* — foregrounds the reframing over the application.
3. *Solving systems of equations with distributed probabilistic numerics* — the most concrete.

## The one-sentence thesis

Classical iterative solvers are message passing with the uncertainty deleted — and when you put the
uncertainty back, it turns out to describe the *operator*, not the *computation*. Closing that gap is
what probabilistic numerics has to offer a parallel PDE solver.

## The two-notebook structure

The tutorial is deliberately split into a *reframing* and a *payoff*, because notebook 1 alone does not
answer "why bother":

* **Notebook 1 — `01-linear-systems-by-message-passing.py`.** The dictionary
  (matrix ↔ precision, sparsity ↔ conditional independence), GaBP in 30 lines, Jacobi = GaBP with the
  second moment deleted. Ends honestly: on one laptop, CG wins on iterations and the BP variances are
  over-confident.
* **Notebook 2 — `02-domain-decomposition-as-message-passing.py`.** Regroup the unknowns by subdomain
  and the same construction becomes a parallel PDE solver. Domain decomposition *is* message passing
  (the message is a Schur complement); granularity is a dial from direct solver to scalar BP; and the
  belief turns out to be blind to its own numerical error.

**Do not present notebook 1 without notebook 2.** The first raises the "so what?" that the second
answers. If time is short, cut depth from notebook 1's §4.8–4.10, not notebook 2.

**An earlier draft of notebook 2 used a robot-web localisation problem** (now in `archive/python/`). It
produced larger headline numbers — 22× on incremental updates, graceful degradation under packet loss —
but a ProbNum audience would correctly object that its covariance is a *statistical* posterior about
measurement noise, i.e. distributed estimation rather than numerics. The PDE version trades showier
numbers for a defensible claim. Keep the archived one only as a source of application anecdotes.

## Narrative arc (matching the notebook sections)

| § | Beat | Time |
|---|------|------|
| 1 | **Problem specification.** $Ax=b$, sparse, large. Three facts about the regime: $A$ is a graph, the data is already distributed, global synchronisation is the bottleneck. Show the lattice-as-graph figure. | 10 min |
| 2 | **Classical numerical approach.** Direct (fill-in as graph densification), Jacobi/Gauss–Seidel (already local!), CG (fast, but two all-reduces per step). All return a number. | 10 min |
| 3 | **Probabilistic numerical approach.** Two readings: the familiar Krylov/BayesCG one (dense covariance, global policy), and the one we use — $p(x) \propto e^{-\frac12 x^\top Ax + b^\top x} = \mathcal{N}(A^{-1}b, A^{-1})$. The dictionary table: matrix ↔ precision, sparsity ↔ conditional independence, solution ↔ marginal means, $\mathrm{diag}(A^{-1})$ ↔ marginal variances, $1/A_{ii}$ ↔ *conditional* variance. | 15 min |
| 4 | **Message-passing version.** Factor graph → two-scalar messages → 30 lines of code. Then, in order: 3×3 sanity check; trees (= Gaussian elimination, exact variances); *what a belief means before convergence* (the $k$-hop computation tree; the information front animation); loops (means exact, variances over-confident); **the punchline** (Jacobi = GaBP with $P_{ij}:=0$); scheduling (flooding vs serial, no barriers); scaling (round count saturates when the correlation length is finite); failure modes. | 40 min |
| 5 | **Where this goes.** Non-symmetric systems, GaBP as a multigrid smoother, interior-point LP, calibrated variances via generalised BP. Hand over to notebook 2. | 5 min |

### Notebook 2 — domain decomposition (≈ 40 min)

| § | Beat | Time |
|---|------|------|
| 1 | **The problem and the three solutions.** $(c-\Delta)u=f$ with a manufactured solution, so we can see all three of $u$, $u_h$, $\mu^{(k)}$ — and therefore both the discretisation and the algebraic error. Set this up carefully; §8 depends on it. | 5 min |
| 2 | **Decomposition = a coarser factor graph.** One node per rank; edges are the halo exchanges the code already does. | 4 min |
| 3 | **The message is a Schur complement.** Two subdomains: $A_{11} + \Lambda_{0\to1} = S$, to machine zero, and the message has rank = interface size — the discrete Dirichlet-to-Neumann map. Chain of $P$ subdomains → exact in $P-1$ rounds = block substructuring. *"Substructuring was always message passing; nobody wrote it that way."* | 8 min |
| 4 | **Granularity is a dial.** One subdomain = direct solve, exact variances. One node = scalar BP, most bias. Coarse blocks absorb their internal loops exactly — cluster variation from the numerics side. | 7 min |
| 5 | **The parallel cost model.** GaBP round = 1 halo exchange; CG iteration = 1 halo + $\log_2 P$ all-reduce. Then the straggler sweep — **and say plainly that GaBP degrades too**; the win is a constant factor from avoiding the barrier, not an asymptotic one. | 6 min |
| 6 | **What is the covariance, with no noise?** $A^{-1}$ is the discrete Green's function; by Lindgren–Rue–Lindström it is a Matérn variance. Satisfying — and it is a property of the *operator*, identical however you solve. | 5 min |
| 7 | **Blindness.** The precision recursion has no $b$ in it. Same operator, two right-hand sides, errors differing several-fold, **bit-identical** reported uncertainty. | 3 min |
| 8 | **When should you stop?** The right round is where the algebraic error crosses the discretisation error; past it, every round is wasted. The belief is a flat line through the whole decision. State the open problem: walk-sum truncation bounds, message residuals, probabilistic discretisation error. | 4 min |

Total ≈ 85 min plus questions; drop notebook 1's §4.8–4.10 first if running long, then notebook 2's §5; never notebook 2's §7–8.

## The five moments that must land

1. **The dictionary** (nb1 §3). If the audience leaves with "the matrix is a precision matrix and its
   sparsity is a conditional independence graph", everything else follows.
2. **The belief before convergence** (nb1 §4.5). The $k$-round belief is the *exact* posterior of the
   $k$-hop computation tree. State it precisely and **do not call it an error bar** — nb2 §7 shows it is
   not one. The interesting content is the gap between "anytime belief" and "anytime error estimate".
3. **Jacobi = GaBP minus the second moment** (nb1 §4.7, verified at $6.7\times10^{-16}$). This is the
   line people will repeat afterwards.
4. **The message is a Schur complement** (nb2 §3), checked to machine zero, with a chain of subdomains
   converging in exactly $P-1$ rounds. *"Substructuring was always message passing."* This is the line
   that earns the audience's attention, because it is about a method they already use.
5. **Blindness, and the stopping question** (nb2 §7–8). Two right-hand sides, errors differing
   several-fold, identical reported uncertainty — because the precision recursion has no $b$ in it. Then:
   the one thing a PDE solver needs to know is when to stop, and the belief is silent. **If only one
   slide survives, this is it** — it is the open problem the tutorial hands to the room.

## The question this tutorial must survive

> *"Fine, but is there any actual advantage to belief propagation over the standard probnum solver?"*

Notebook 1 alone cannot answer this and should not pretend to. The answer has three parts, in this order:

1. **Not on iterations, and not on variances.** Say this first and unprompted. CG converges in fewer
   iterations; converged loopy BP variances are over-confident; and if you *can* centralise, selected
   inversion (Takahashi / Rue & Martino) computes $\mathrm{diag}(A^{-1})$ exactly and fast.
2. **On cost structure.** BayesCG's belief is a dense $n \times n$ covariance and every iteration needs
   two global reductions. GaBP's belief is $O(\mathrm{nnz})$ and every update is nearest-neighbour. That
   is a scaling statement, not a constant factor.
3. **On what it reveals.** This is the answer to lead with now. Writing domain decomposition as message
   passing costs nothing and makes a real gap visible: these solvers already carry a belief, the belief
   is exactly the operator's Green's function, and it says nothing about the error of the computation
   that produced it — not the algebraic error, not the discretisation error. The contribution is not a
   faster solver; it is a well-posed probnum problem sitting inside a method thousands of people run.
   (The measured *speed* wins in the literature — IPU bundle adjustment, FMGaBP, Robot Web,
   power-grid state estimation — are worth one slide as evidence that people deploy this, but none of
   them are "BP converges faster", so do not lean on them.)

## Honest caveats to state out loud (do not let a questioner find them first)

* **GaBP is not faster than CG in iterations.** Never was, never will be. Say it unprompted.
* Converged GaBP variances are **wrong** on loopy graphs — over-confident. In nb1's lattice the ratio is
  0.91–0.99; scalar BP on nb2's grid reaches 0.926.
* **If you can centralise, selected inversion wins** for $\mathrm{diag}(A^{-1})$ (Takahashi; Rue &
  Martino). GaBP does not win on variances. Concede this early; someone in a ProbNum room will know it.
* Convergence is not guaranteed in general; the empirical basin is much wider than diagonal dominance or
  walk-summability, but it ends roughly where $A$ stops being positive definite, and damping does not
  rescue the indefinite case. (In nb2's decomposition it is guaranteed — the system is an SPD
  generalised Laplacian.)
* **The parallel cost model is a model.** $\log_2 P$ for an all-reduce is a fair textbook figure, but
  real machines have communication-avoiding and pipelined Krylov variants specifically designed to hide
  it. The honest claim is a constant factor, not a change of asymptotics.
* **GaBP is not immune to stragglers either.** Information that must pass *through* a slow rank is still
  delayed, and the round count grows roughly linearly in the slowdown. What it avoids is the barrier.
* **Plain GaBP is not a competitive domain-decomposition method.** Classical DD has optimised
  transmission conditions, coarse-grid corrections and condition-number bounds; GaBP as presented has
  none. The claim is that DD *is* message passing, not that this is a better DD.
* Everything assumes $A = A^\top$ positive definite, and the PDE is linear.

## Numbers for the slides

### Notebook 1

| claim | value |
|---|---|
| chain (tree), 80 unknowns | means and variances exact to $2\times10^{-13}$ / $0$, 43 rounds |
| 24×24 lattice, screening 0.4 | means exact to $10^{-12}$; BP/true variance ratio 0.909–0.987 |
| Jacobi ≡ GaBP with $P_{ij}:=0$ | agree to $6.7\times10^{-16}$ over 40 rounds |
| serial vs flooding schedule | 67 vs 112 rounds to $10^{-10}$ |
| rounds to $10^{-8}$, $n = 64 \to 4096$ | $c=2$: 22 → 30 · $c=0.4$: 55 → 104 · $c=0$: 124 → 6810 |

### Notebook 2 (domain decomposition)

All verified from a full run on 2026-08-14, after the manufactured solution was corrected.

**Exact structural results** (these cannot drift):

| claim | value |
|---|---|
| two subdomains: $A_{11} + \Lambda_{0\to1}$ vs the Schur complement $S$ | **0.00e+00** — identical |
| rank of the message | **16** on 128-dof subdomains = the interface — the DtN map |
| chain of $P$ subdomains | exact in **exactly $P-1$ rounds**: 2→1, 3→2, 4→3, 6→5, 8→7, 12→11 |
| blindness: two right-hand sides, one operator | reported std differs by **0.00e+00** |

**Measured results** (32×32 grid for granularity, 64×64 on 64 ranks for the rest):

| claim | value |
|---|---|
| granularity: 1 subdomain | 1 round, variances **exact** (it is a direct solve) |
| granularity: 4 / 16 / 64 / 256 subdomains | 21 / 21 / 24 / **41** rounds; variance ratio 0.9881 → **0.9832** |
| cost model | GaBP **26** rounds vs CG **58** iters × $(1+\log_2 64)$ = **406** → **15.6×** |
| straggler, $S$ = 1 / 10 / 50 | GaBP 26 / 70 / 273 rounds; CG 406 / 928 / 3248 → **15.6× / 13.3× / 11.9×** |
| blindness: actual errors, smooth vs rough $b$ | differ by 1.5×–4.5× at every round |
| stopping | discretisation error **1.621e-05**; algebraic error crosses it at round **14**; total error is smallest at round **14** and flat thereafter; the solver ran **26** — about **46% wasted** |

Three quotable lines:

* **"The message a subdomain sends is its Schur complement. Substructuring was always message passing."**
* **"Two right-hand sides, errors differing four-fold, and the solver reports the same uncertainty to the
  last bit — because the precision recursion never sees $b$."**
* **"It should have stopped at round 14. It ran 26. Its belief reports the same number at both."**

Note the straggler advantage is roughly **constant at 10–15×**, not growing — both methods degrade. That
is the honest reading and the notebook says so.

**A trap worth knowing about.** The manufactured solution must not be $\sin \pi x \sin \pi y$: it is an
exact eigenvector of the five-point stencil, so CG converges in **one** iteration and the cost comparison
is meaningless. (This was caught only because a reported "CG: 1 iteration" looked impossible.) A
polynomial like $x(1-x)y(1-y)$ fails the other way — the stencil is exact for cubics, so the
discretisation error is $10^{-16}$ and §8 has nothing to show. The notebook uses
$\sin(\pi x)\sin(\pi y)e^{x+y}$ and says why. Good answer to have ready if anyone asks.

## Rehearsal checklist

* [ ] Run 1: solo, timed, no interruptions. Does notebook 1's §4 fit in 40 minutes, and notebook 2 in 40?
* [ ] Run 2: with a colleague who does *not* know belief propagation. Where do they get lost — the
      message derivation, or the computation-tree argument?
* [ ] Run 3: with a colleague who does *not* know probabilistic numerics. Does the nb1 §3 dictionary land?
* [ ] Run 4: **have someone play the hostile questioner** and ask "so what does BP buy me?" after
      notebook 1. Rehearse the three-part answer above until it is reflexive.
* [ ] Check the interactive cells on the presentation machine (marimo + plotly at projector resolution,
      light theme, font sizes).
* [ ] Live-code vs pre-run: notebook 1 restarts in ~10 s, so live is safe. **Notebook 2's first load runs
      the granularity and straggler sweeps and takes a few minutes — open it before the session and
      leave it warm.**
* [ ] Re-read notebook 2's numeric tables into the slides after any parameter change.

## Open TODOs

* [ ] Pick the final title.
* [ ] **Fill the notebook-2 numbers table from a live run** and put them on slides.
* [ ] Optional Julia/RxInfer edition of the message-passing section, mirroring the archived notebooks —
      RxInfer gives the factor graph for free and would make the "this is just inference" point visually.
* [ ] Consider a slide-only intro (5 min) before the notebook, so the first thing on screen is not code.
* [ ] **The strongest missing figure**: a coarse-grid correction added as one extra factor-graph node
      (nb2 exercise 1). If it makes the round count $P$-independent, that is two-level domain
      decomposition *derived* as message passing, and it would be the best single result in the tutorial.
      Worth attempting before the conference.
* [ ] Decide whether the variance over-confidence gets a fix-it demo (generalised BP / cluster variation,
      Fanaskov's second algorithm) — the granularity sweep already gestures at it — or stays open.
* [ ] Consider showing the non-symmetric extension (Shental §VII augmented system, or Fanaskov's
      non-symmetric messages). Currently only in notebook 1's outlook.
* [ ] Sanity-check the $\log_2 P$ all-reduce model against someone who does HPC for a living; a
      practitioner in the room will have opinions, and it is better to have the caveat pre-loaded.
