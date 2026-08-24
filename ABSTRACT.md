# Title and abstract — ProbNum 2026 tutorial

## Title

**Solving systems of equations with distributed probabilistic numerics**

## Abstract (~170 words)

*How hot is this chip?* A silicon die typically has a few thousand compute tiles, each drawing power. 
Cooling control needs to calculate the whole steady-state temperature field before the die gets too hot. 
Figuring out how hot each part is, is equivalent to solving a large sparse linear system in a rush.

Classical linear solvers are excellent tools, but consume large amounts of power to provide exact answers at scale. 
But the cooler doesn't need an exact temperature, a fast approximation is much better. Probabilistic linear solvers 
do exactly that: their accuracy can be tailored to the computation budget. Probabilistic numerics by message passing
takes that one step further and distributes computation itself according to the sparsity pattern of the matrix. 
In this tutorial, we pose the classical problem of solving a linear system of equations, and briefly discuss the classical
conjugate gradient and Krylov methods. We then present a state-of-the-art probabilistic numerical solver as well as its 
message passing variant. Solving the system becomes marginal inference in a Gauss Markov random field, which is local, 
asynchronous, and comes with per-node uncertainty quantification. We provide Marimo (Python) and Pluto (Julia) notebooks 
for participants to explore and familiarize themselves with these concepts. We are looking forward to a fun interactive session.

## One-line hook

> The Jacobi method is Gaussian belief propagation without the second moment.

## Practicalities (fill in as needed for the site)

* **Format.** ~85 minutes plus questions; a single executable notebook (marimo/Python, with a
  Julia/Pluto edition), run live end-to-end in about ten seconds.
* **Prerequisites.** Linear algebra and basic Gaussians. No prior exposure to graphical models or to
  belief propagation is assumed; both are derived from scratch.
* **Takeaways.** The precision-matrix dictionary; Gaussian belief propagation as a solver in ~30 lines;
  what an anytime belief does and does not certify; and two open problems (calibrated local variances,
  and beliefs about the computation rather than the operator).
* **Materials.** <https://github.com/biaslab/ProbNum2026-Tutorial>
