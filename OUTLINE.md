# Tutorial outline — notebook 01, *Linear systems by message passing*

**Scope.** This outline covers `python/01-linear-systems-by-message-passing.py` only — the notebook that
will be presented. `02-domain-decomposition-as-message-passing.py` is kept as a backup and is not part of
the talk. An earlier version of this outline planned a two-notebook session in which notebook 2 supplied
the payoff; **that support is gone**, so the "why bother?" question now lands squarely on notebook 1 and
the answer has to be carried by the speaker. See *The question this tutorial must survive*, below. It is
the single most important section of this document.

Working title, in order of preference:

1. **Probabilistic numerics at scale by distributed inference** — names both the payoff (scale) and the mechanism. Best if the tutorial is pitched as *"here is what probnum is missing"*.
2. *A distributed inference perspective to probabilistic numerics* — foregrounds the reframing over the application.
3. *Solving systems of equations with distributed probabilistic numerics* — the most concrete.

## The one-sentence thesis

Classical iterative solvers are message passing with the uncertainty deleted — and when you put the
uncertainty back, it turns out to describe the *operator*, not the *computation*. Closing that gap is
what probabilistic numerics has to offer a parallel solver.

## Narrative arc

| § | Beat | Time |
|---|------|------|
| 1 | **Problem specification.** The story: a tiled accelerator whose control loop needs the steady-state temperature of every tile, faster than the die's thermal time constant. Heat to coolant + heat to neighbours = power dissipated *is* the five-point stencil for $(c-\Delta)u=f$. Three facts about the regime, each anchored in the die: $A$ **is** the floorplan; $b_i$ was measured by tile $i$ and never assembled anywhere; a barrier across a few thousand tiles costs more than the arithmetic between barriers. Chain (one row of tiles, a tree) and lattice (the die, loopy). $c$ = coolant coupling = how far a hotspot is felt = correlation length. Show the lattice-as-graph figure. | 12 min |
| 2 | **Classical numerical approach.** Direct (fill-in as graph densification), Jacobi/Gauss–Seidel (already local!), CG (fast, but two all-reduces per step). All three return a vector plus an error bound in quantities as unknown as the solution. | 8 min |
| 3 | **Probabilistic numerical approach.** Two readings: the familiar Krylov/BayesCG one (dense covariance, global policy), and the one we use — $p(x) \propto e^{-\frac12 x^\top Ax + b^\top x} = \mathcal{N}(A^{-1}b, A^{-1})$. The dictionary table: matrix ↔ precision, sparsity ↔ conditional independence, solution ↔ marginal means, $\mathrm{diag}(A^{-1})$ ↔ marginal variances, $1/A_{ii}$ ↔ *conditional* variance. Then the payoff of §1: $(A^{-1})_{ij}$ is the chip's thermal impedance, so the marginal variance is a quantity a thermal engineer would have measured — and $1/A_{ii}$ is the same number with the neighbours pinned to ambient. | 15 min |
| 4 | **Message-passing version.** See the sub-budget below. | 40 min |
| 5 | **Where this goes.** Non-symmetric systems and GaBP as a multigrid smoother; interior-point LP; the two open uncertainty problems (calibration, and belief-vs-error); applications where the factor graph is the physical machine, with the graph-processor number as evidence that this pays in wall-clock. | 10 min |

Total ≈ 85 min plus questions. **This budget assumes the slot that was previously going to hold two
notebooks.** If the slot is shorter, cut §4.10 then §4.8; if it is longer (a 2-hour tutorial), the honest
expansions are a live derivation of the §4.2 messages on the board and more time on §4.5, not more demos.

### §4 sub-budget

| § | Beat | Time |
|---|------|------|
| 4.1 | Factor graph: one factor per node, one per edge; every quantity in it is already owned by node $i$ | 3 |
| 4.2 | The messages: two scalars per directed edge, and the sign — $P_{ij} = -A_{ij}^2/P_{i\setminus j} < 0$, "you are less certain than you thought". No $n$ anywhere in the update | 6 |
| 4.3 | $3\times3$ sanity check — symmetric but **indefinite**, and still exact | 2 |
| 4.4 | Trees: the message sweep *is* Gaussian elimination; the chain gives exact variances off a dense inverse we never formed | 4 |
| 4.5 | **What a belief means before convergence** — the $k$-hop computation tree, the information front on the lattice, *and* the box that says this is not an error bar | 8 |
| 4.6 | Loops: means exact, variances over-confident | 3 |
| 4.7 | **The punchline** — Jacobi = GaBP with $P_{ij} := 0$ | 5 |
| 4.8 | Scheduling: flooding vs serial, no barriers either way | 3 |
| 4.9 | Scaling: the round count saturates exactly when the correlation length is finite | 4 |
| 4.10 | Failure modes (cut first if running long) | 2 |

## The three moments that must land

1. **The dictionary** (§3). If the audience leaves with "the matrix is a precision matrix and its sparsity
   is a conditional independence graph", everything else follows. The thermal-impedance reading is what
   makes the last row of the table feel like a fact rather than a definition.
2. **The belief before convergence — and its ceiling** (§4.5). The $k$-round belief is the *exact*
   posterior of the $k$-hop computation tree. State it precisely and **do not call it an error bar**: the
   precision recursion $P_{ij} = -A_{ij}^2/P_{i\setminus j}$ contains no $b$, so the same operator with a
   different right-hand side gives bit-identical uncertainty and completely different errors. The
   interesting content is the gap between "anytime belief" and "anytime error estimate". Deliver both
   halves or the second one arrives as a hostile question.
3. **Jacobi = GaBP minus the second moment** (§4.7, verified at $6.7\times10^{-16}$). This is the line
   people will repeat afterwards.

## The question this tutorial must survive

> *"Fine, but is there any actual advantage to belief propagation over the standard probnum solver?"*

Notebook 1 cannot answer this with a benchmark and should not pretend to. Answer in three parts, in this
order, and volunteer part 1 before anyone asks:

1. **Not on iterations, and not on variances.** CG converges in fewer iterations (41 against 67 and 112
   here); converged loopy BP variances are over-confident; and if you *can* centralise, selected inversion
   (Takahashi; Rue & Martino) computes $\mathrm{diag}(A^{-1})$ exactly and fast.
2. **On cost structure.** BayesCG's belief is a dense $n \times n$ covariance and every iteration needs two
   global reductions. GaBP's belief is $O(\mathrm{nnz})$ and every update is nearest-neighbour. That is a
   scaling statement, not a constant factor — and where the hardware agrees, it shows up as wall-clock:
   Ortiz et al. (2020) solve bundle adjustment on 1216 graph-processor cores in <40 ms against 1450 ms for
   a CPU library; FMGaBP beats parallel MG-PCG by 2.9× on matrix-free FEM. Neither is "BP converges
   faster", so do not lean on them for iteration counts.
3. **On what it reveals.** Lead with this. The reframing costs nothing and makes a real gap visible:
   these solvers already carry a belief; the belief is exactly the operator's Green's function
   ($\mathrm{diag}(A^{-1})$, the thermal impedance of §1); and it says nothing about the error of the
   computation that produced it. The contribution is not a faster solver — it is a well-posed
   probabilistic-numerics problem sitting inside a method that thousands of people already run.

## Honest caveats to state out loud (do not let a questioner find them first)

* **The belief is not an error bar.** $P_{ij} = -A_{ij}^2/P_{i\setminus j}$ never sees $b$: change the
  right-hand side and every variance in the notebook is unchanged to the last bit while every error is
  different. Anytime *belief* $\neq$ anytime *error estimate*. Say this before §4.5's animation seduces
  anyone, including us.
* **GaBP is not faster than CG in iterations.** Never was, never will be. Say it unprompted.
* **Converged GaBP variances are wrong on loopy graphs** — over-confident, ratio 0.909–0.987 on our
  lattice. The means are exact.
* **If you can centralise, selected inversion wins** for $\mathrm{diag}(A^{-1})$ (Takahashi; Rue &
  Martino). GaBP does not win on variances. Concede this early; someone in a ProbNum room will know it.
* **The round count is only $n$-independent when the correlation length is finite.** On the unscreened
  Laplacian it grows with the diameter, exactly as unpreconditioned classical methods do — and that says
  precisely what a preconditioner must do: shorten the correlation length.
* **The parallel cost model is a model.** Counting an all-reduce as $\log_2 P$ is a fair textbook figure,
  but real machines have communication-avoiding and pipelined Krylov variants designed specifically to
  hide it. The honest claim is a constant factor, not a change of asymptotics.
* **GaBP is not immune to stragglers either.** Information that must pass *through* a slow node is still
  delayed. What it avoids is the barrier, not the latency.
* **Convergence is not guaranteed**, and the sufficient conditions are far from tight: on the frustrated
  lattice, diagonal dominance fails at $w = 0.25$ and walk-summability at $w \approx 0.26$, and *neither
  failure costs anything* — the solver still converges at $w = 0.29$ (297 rounds) and breaks between 0.29
  and 0.30, essentially where $A$ stops being positive definite. Damping smooths the borderline regime and
  does not rescue the indefinite case.
* Everything here assumes $A = A^\top$ positive definite. Non-symmetric is §5's first bullet, not a result.

## Numbers from the notebook (verified against a full run)

| claim | value |
|---|---|
| chain (tree), 80 unknowns | means exact to $2.1\times10^{-13}$, variances exact to $0$, 43 rounds |
| 24×24 lattice, screening 0.4 | means exact to $1.2\times10^{-12}$ in 136 rounds; BP/true variance ratio 0.909–0.987 |
| Jacobi ≡ GaBP with $P_{ij}:=0$ | agree to $6.7\times10^{-16}$ over 40 rounds |
| what the deleted second moment was worth | residual after 40 rounds: Jacobi $9.0\times10^{-3}$, GaBP $1.9\times10^{-4}$ |
| rounds to $10^{-10}$, 24×24 lattice | CG 41 · GaBP serial 67 · Gauss–Seidel 110 · GaBP flooding 112 · Jacobi >120 |
| rounds to $10^{-8}$, $n = 64 \to 4096$ | $c=2$: 22 → 30 · $c=0.4$: 55 → 104 · $c=0$: 124 → 6810 |
| frustrated lattice, $12\times12$ | dominance lost at $w=0.25$, walk-summability at $w=0.26$, still converging at $w=0.29$ (297 rounds), diverges at $w=0.30$ ($\lambda_{\min}=+0.007$) |

Two quotable lines:

* **"The Jacobi method is Gaussian belief propagation with the second moment deleted — to sixteen digits."**
* **"Change the right-hand side and every error changes; every reported uncertainty does not move a bit.
  The belief is about the operator, not about the computation."**

## Rehearsal checklist

* [ ] Run 1: solo, timed, no interruptions. Does §4 fit in 40 minutes?
* [ ] Run 2: with a colleague who does *not* know belief propagation. Where do they get lost — the message
      derivation (§4.2), or the computation-tree argument (§4.5)?
* [ ] Run 3: with a colleague who does *not* know probabilistic numerics. Does the §3 dictionary land, and
      does the thermal-impedance reading help or distract?
* [ ] Run 4: **have someone play the hostile questioner** and ask "so what does BP buy me?" Rehearse the
      three-part answer above until it is reflexive. This matters more now that notebook 2 is not there to
      answer it.
* [ ] Check the interactive cells on the presentation machine (marimo + plotly at projector resolution,
      light theme, font sizes).
* [ ] Decide live-code vs pre-run: the notebook runs end-to-end in ~10 s, so a live restart is safe.
* [ ] Rehearse the §4.10 slider — know which $w$ to type to show convergence at 0.29 and divergence at 0.30.

## Open TODOs

* [ ] Pick the final title.
* [ ] Consider a slide-only intro (5 min) so the first thing on screen is not code — the §1 story is
      written to work as spoken narration over a picture of a die.
* [ ] Decide whether §4.9's scaling table is shown as a live run (slow at $c=0$, $n=4096$) or pre-computed.
* [ ] Decide whether the variance over-confidence gets a fix-it demo (generalised BP / cluster variation,
      Fanaskov's second algorithm) or stays an open problem.
* [ ] Consider showing the non-symmetric extension (Shental §VII augmented system, or Fanaskov's
      non-symmetric messages). Currently only in §5's outlook.
* [ ] Sanity-check the $\log_2 P$ all-reduce model with someone who does HPC for a living; a practitioner
      in the room will have opinions, and it is better to have the caveat pre-loaded.
* [ ] Optional Julia/RxInfer edition of the message-passing section — RxInfer gives the factor graph for
      free and would make the "this is just inference" point visually.
