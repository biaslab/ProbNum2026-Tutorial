# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""Krylov methods and conjugate gradients.

ProbNum 2026 tutorial — companion notebook to 01-linear-systems-by-message-passing.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import plotly.graph_objects as go
    return go, mo, np, sp, spla


@app.cell
def _():
    PAL = dict(blue="#2a78d6", black="#0b0b0b", green="#008300", orange="#eb6834",
               pink="#e87ba4", gray="#898781", violet="#4a3aa7", white="#fcfcfb")

    def base_layout(fig, title="", xlabel="", ylabel="", **kw):
        fig.update_layout(template="plotly_white", title=title,
                          xaxis_title=xlabel, yaxis_title=ylabel,
                          margin=dict(l=60, r=20, t=50, b=50), **kw)
        return fig
    return PAL, base_layout


@app.cell
def _(mo):
    mo.md(
        r"""
    # Krylov methods and conjugate gradients

    **ProbNum 2026 tutorial — companion to notebook 01**

    Jacobi and Gauss–Seidel (notebooks 01a, 01b) both have the shape

    $$
    x^{(t+1)} = x^{(t)} + N^{-1} r^{(t)},
    $$

    with $N$ a fixed, cheap approximation to $A$ — the diagonal, or the lower triangle. The step length is decided in advance and never revisited. Expand the recursion and you find that $x^{(t)}$ is always some fixed polynomial in $A$ applied to $b$:

    $$
    x^{(t)} \in \operatorname{span}\{b,\; Ab,\; A^2b,\; \dots,\; A^{t-1}b\} \;=\; \mathcal{K}_t(A, b),
    $$

    the **Krylov subspace**. The stationary methods live in it with coefficients chosen ahead of time from a spectral-radius argument.

    Krylov methods ask the obvious next question: *given that we are stuck in this subspace, why not pick the best point in it?* Conjugate gradients does exactly that, and the price of "best" is precisely the global communication that notebook 01 sets out to avoid.
    """
    )
    return


@app.cell
def _(np, sp):
    def grid_matrix(m, screening=0.0):
        "Five-point stencil for (c − Δ) on an m×m lattice."
        d = 4.0 + screening
        T = sp.diags([-np.ones(m - 1), d * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
        band = sp.diags([-np.ones(m - 1), -np.ones(m - 1)], [-1, 1])
        return (sp.kron(sp.eye(m), T) + sp.kron(band, sp.eye(m))).tocsr()

    def bump_forcing(m, centers=((0.3, 0.35), (0.7, 0.65)), width=0.12):
        "A hot core and a cooling channel on the m×m lattice."
        g = (np.arange(m) + 0.5) / m
        X, Y = np.meshgrid(g, g, indexing="ij")
        f = np.zeros((m, m))
        for k, (cx, cy) in enumerate(centers):
            f += (1.0 if k % 2 == 0 else -0.8) * np.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
        return f.ravel()
    return bump_forcing, grid_matrix


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. What "best" means

    For symmetric positive definite $A$ there is a natural notion of best. Define the **energy norm** $\|v\|_A = \sqrt{v^\top A v}$; then

    $$
    \tfrac12 x^\top A x - b^\top x \;=\; \tfrac12\|x - x_\ast\|_A^2 \;-\; \tfrac12 \|x_\ast\|_A^2 ,
    $$

    so minimising the quadratic $q(x)$ and minimising the energy-norm error are the same thing. (This is the same $q$ whose exponential is the Gaussian of notebook 01 §3 — the objective classical numerics minimises is the negative log-density probabilistic numerics conditions on.)

    Conjugate gradients returns

    $$
    \boxed{\;x_k \;=\; \operatorname*{arg\,min}_{x \,\in\, x_0 + \mathcal{K}_k(A, r_0)} \|x - x_\ast\|_A\;}
    $$

    — the *exact* minimiser over a $k$-dimensional subspace, found with one matrix–vector product per step and three vectors of storage. The trick is that the residuals come out mutually orthogonal and the search directions $A$-orthogonal ("conjugate"), which collapses what should be a $k$-dimensional least-squares problem into a three-term recurrence.

    Two consequences fall straight out of the definition.

    * **It terminates.** After $n$ steps the subspace is all of $\mathbb{R}^n$, so $x_n = x_\ast$ exactly. In floating point this is not something to rely on, but it does mean CG is not really an iterative method with a rate — it is a direct method one usually stops early.
    * **It cannot be beaten in its own subspace.** Any method whose iterates lie in $\mathcal{K}_k$ — including Jacobi, Gauss–Seidel and, on a fixed schedule, message passing — has energy-norm error at least CG's. That is a strong statement, and it is why "CG wins on iterations" in notebook 01 is not a contingent fact about that example.
    """
    )
    return


@app.cell
def _(np):
    def conjugate_gradients(A, b, iters, x0=None):
        "Textbook CG. The two dot products per step are the whole story of §4."
        x = np.zeros_like(b, dtype=float) if x0 is None else x0.astype(float).copy()
        r = b - A @ x
        p = r.copy()
        rr = r @ r                     # global reduction #1
        out = [x.copy()]
        for _ in range(iters):
            Ap = A @ p                 # local: one halo exchange
            alpha = rr / (p @ Ap)      # global reduction #2
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = r @ r             # global reduction #1 of the next step
            p = r + (rr_new / rr) * p
            rr = rr_new
            out.append(x.copy())
        return out
    return (conjugate_gradients,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. The convergence bound, and why it is pessimistic

    Because $x_k$ is optimal over all polynomials of degree $k$ with $p(0)=1$, the error obeys

    $$
    \frac{\|x_k - x_\ast\|_A}{\|x_0 - x_\ast\|_A}
    \;\le\; \min_{\substack{p \in \mathbb{P}_k \\ p(0) = 1}} \; \max_{\lambda \in \Lambda(A)} |p(\lambda)| .
    $$

    Replacing the discrete spectrum $\Lambda(A)$ by the interval $[\lambda_{\min}, \lambda_{\max}]$ and using Chebyshev polynomials gives the textbook bound in terms of the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$:

    $$
    \frac{\|x_k - x_\ast\|_A}{\|x_0 - x_\ast\|_A} \;\le\; 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{k} .
    $$

    Compare with the stationary methods, whose rate is $\rho \approx 1 - O(1/\kappa)$: CG's is $1 - O(1/\sqrt{\kappa})$. Squaring the condition number's exponent is the entire reason Krylov methods took over computational science.

    But that bound throws away everything about the spectrum except its two endpoints, and §3 shows how much it is throwing away.
    """
    )
    return


@app.cell
def _(bump_forcing, conjugate_gradients, grid_matrix, np):
    m_grid = 24
    c_grid = 0.4
    A_grid = grid_matrix(m_grid, screening=c_grid)
    b_grid = bump_forcing(m_grid)
    Ad_grid = A_grid.toarray()
    eig_grid = np.linalg.eigvalsh(Ad_grid)
    x_star = np.linalg.solve(Ad_grid, b_grid)
    kappa_grid = eig_grid[-1] / eig_grid[0]

    def a_norm_errors(A, xs, x_star_):
        e0 = None
        out = []
        for x in xs:
            e = x - x_star_
            val = float(np.sqrt(max(e @ (A @ e), 0.0)))
            e0 = val if e0 is None else e0
            out.append(val / e0)
        return np.array(out)

    cg_iterates = conjugate_gradients(A_grid, b_grid, 120)
    cg_err = a_norm_errors(A_grid, cg_iterates, x_star)
    return (
        A_grid,
        Ad_grid,
        a_norm_errors,
        b_grid,
        c_grid,
        cg_err,
        eig_grid,
        kappa_grid,
        m_grid,
        x_star,
    )


@app.cell
def _(PAL, base_layout, cg_err, go, kappa_grid, np):
    _k = np.arange(len(cg_err))
    _bound = 2 * ((np.sqrt(kappa_grid) - 1) / (np.sqrt(kappa_grid) + 1)) ** _k

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_k, y=np.maximum(_bound, 1e-18), mode="lines",
                              name="Chebyshev bound  2((√κ−1)/(√κ+1))ᵏ",
                              line=dict(color=PAL["orange"], width=2, dash="dash")))
    _fig.add_trace(go.Scatter(x=_k, y=np.maximum(cg_err, 1e-18), mode="lines",
                              name="actual CG energy-norm error",
                              line=dict(color=PAL["blue"], width=2)))
    base_layout(_fig, title="CG against its own bound",
                xlabel="iteration k", ylabel="‖xₖ − x⋆‖_A / ‖x₀ − x⋆‖_A",
                legend=dict(x=0.02, y=0.02, yanchor="bottom"))
    _fig.update_yaxes(type="log", range=[-16, 0.5])
    _fig.update_xaxes(range=[0, 80])
    _fig.update_layout(height=420)
    _fig
    return


@app.cell
def _(cg_err, eig_grid, kappa_grid, mo, np):
    _hit = int(np.argmax(cg_err < 1e-10)) if (cg_err < 1e-10).any() else None
    _rate = (np.sqrt(kappa_grid) - 1) / (np.sqrt(kappa_grid) + 1)
    _pred = int(np.ceil(np.log(1e-10 / 2) / np.log(_rate)))
    mo.md(
        f"""
    | | value |
    |:--|--:|
    | λ_min, λ_max | {eig_grid[0]:.4f}, {eig_grid[-1]:.4f} |
    | condition number κ | {kappa_grid:.3f} |
    | bound's per-step factor (√κ−1)/(√κ+1) | {_rate:.4f} |
    | iterations to 10⁻¹⁰ predicted by the bound | {_pred} |
    | iterations to 10⁻¹⁰ actually taken | {_hit} |

    The bound is honest but loose: it is a statement about the worst spectrum with these two
    endpoints, and our spectrum is much friendlier than that.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. It is the whole spectrum that matters, not the condition number

    The polynomial in the minimax problem has to be small at *every eigenvalue* — and it has $k$ roots to spend. So if the spectrum sits in a few tight clusters, CG places a root in each cluster and is done:

    > A matrix with $\ell$ distinct eigenvalues is solved exactly by CG in **$\ell$ iterations**, whatever the condition number.

    This is why the condition number is a poor predictor in practice, and why preconditioning is best understood as *reshaping the spectrum* rather than *shrinking $\kappa$*.

    Below, three matrices with the **same** condition number $\kappa = 10^4$: eigenvalues spread evenly, gathered into five *exact* clusters, and gathered into five clusters of relative width $10^{-6}$. The exact case behaves as advertised. The third is the warning: a cluster that is tight but not a point needs more than one root, and in floating point the loss of orthogonality among the computed residuals costs more still. Clustering is a powerful effect and a fragile one.
    """
    )
    return


@app.cell
def _(a_norm_errors, conjugate_gradients, np, sp):
    def spectrum_demo(n=200, kappa=1e4, clusters=None, jitter=0.0, seed=0):
        "SPD matrix with a prescribed spectrum; CG only ever sees the eigenvalues."
        rng = np.random.default_rng(seed)
        if clusters is None:
            lam = np.geomspace(1.0, kappa, n)                       # spread out
        else:
            centres = np.geomspace(1.0, kappa, clusters)
            lam = np.repeat(centres, int(np.ceil(n / clusters)))[:n]
            if jitter:
                lam = lam * (1 + jitter * rng.standard_normal(n))   # tight, not exact
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = (Q * lam) @ Q.T
        A = 0.5 * (A + A.T)
        b = rng.standard_normal(n)
        return sp.csr_matrix(A), b, np.sort(lam)

    demo_runs = {}
    for _label, _cl, _jit in (("200 eigenvalues, spread", None, 0.0),
                              ("5 exact clusters", 5, 0.0),
                              ("5 clusters, width 10⁻⁶", 5, 1e-6)):
        _A, _b, _lam = spectrum_demo(clusters=_cl, jitter=_jit)
        _xs = conjugate_gradients(_A, _b, 80)
        _xstar = np.linalg.solve(_A.toarray(), _b)
        demo_runs[_label] = dict(err=a_norm_errors(_A, _xs, _xstar), lam=_lam)
    return (demo_runs,)


@app.cell
def _(PAL, base_layout, demo_runs, go, np):
    _fig = go.Figure()
    for (_label, _d), _col in zip(demo_runs.items(),
                                  (PAL["gray"], PAL["blue"], PAL["orange"])):
        _fig.add_trace(go.Scatter(x=np.arange(len(_d["err"])),
                                  y=np.maximum(_d["err"], 1e-18),
                                  mode="lines", name=_label,
                                  line=dict(color=_col, width=2)))
    base_layout(_fig, title="Same κ = 10⁴, two spectra",
                xlabel="iteration k", ylabel="energy-norm error (relative)",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(type="log", range=[-16, 0.5])
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(demo_runs, mo, np):
    _rows = []
    for _label, _d in demo_runs.items():
        _hit = int(np.argmax(_d["err"] < 1e-10)) if (_d["err"] < 1e-10).any() else ">80"
        _rows.append(f"| {_label} | {_hit} |")
    mo.md("| spectrum | iterations to 10⁻¹⁰ |\n|:--|--:|\n" + "\n".join(_rows) +
          """

Five *exact* clusters take six iterations — for a matrix whose condition number says it should
need hundreds, and nothing about $\\kappa$ changed. Widening those clusters to a relative
$10^{-6}$, which is still far tighter than any real spectrum, already costs a factor of three.
The theory is right and the practice is delicate: this is the gap Greenbaum's analysis of CG in
finite precision is about.
""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. The cost that this tutorial is about

    Here is one CG iteration, annotated by what it needs from the rest of the machine:

    | step | kind | communication |
    |:--|:--|:--|
    | $Ap$ | sparse mat-vec | halo exchange with neighbours |
    | $\alpha = r^\top r / p^\top Ap$ | **inner product** | **all-reduce over all $n$** |
    | $x \leftarrow x + \alpha p$ | vector update | none |
    | $r \leftarrow r - \alpha Ap$ | vector update | none |
    | $\beta = r_{\text{new}}^\top r_{\text{new}} / r^\top r$ | **inner product** | **all-reduce over all $n$** |
    | $p \leftarrow r + \beta p$ | vector update | none |

    **Two all-reduces per iteration.** An all-reduce is a barrier: every processor waits for the slowest, twice per step, forever. On a large machine its latency — not the arithmetic — is what sets the wall-clock time per iteration, and unlike the arithmetic it does not get cheaper as you add processors; it gets more expensive, roughly like $\log P$.

    So the comparison in notebook 01 §4.8 is not "41 CG iterations against 67 GaBP rounds". It is

    $$
    41 \times \bigl(\text{1 halo} + 2\ \text{barriers}\bigr)
    \qquad\text{against}\qquad
    67 \times \bigl(\text{1 halo}\bigr),
    $$

    and which side wins depends entirely on what a barrier costs on your machine. That is the honest form of the argument, and it is why the tutorial's claim is about *cost structure* rather than iteration counts.

    **The fair caveat.** Numerical analysts have not ignored this. *Communication-avoiding* ($s$-step) CG batches $s$ iterations to do one reduction instead of $2s$, and *pipelined* CG overlaps the reduction with the mat-vec so the latency is hidden rather than removed. Both cost numerical stability, and both are more delicate to implement than the six lines above. The message-passing route is a different answer to the same problem: not "make the reduction cheaper", but "never form a quantity that couples all $n$ unknowns".
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Preconditioning, and a trap

    Preconditioning solves $M^{-1}Ax = M^{-1}b$ for some $M \approx A$ that is cheap to invert, and the bound is then governed by the spectrum of $M^{-1}A$.

    Our stencil comes with a trap that is worth meeting once. The obvious cheap choice is the **Jacobi (diagonal) preconditioner** $M = D$ — and for the constant-coefficient five-point stencil $D = (4+c)I$ is a *multiple of the identity*. It rescales every eigenvalue by the same factor, leaves $\kappa$ untouched, and does exactly nothing. Preconditioners are statements about a specific matrix, not general-purpose accelerants.

    Give the tiles different thermal conductances and the picture changes: now $D$ genuinely varies and scaling by it helps. Below, both cases.
    """
    )
    return


@app.cell
def _(np, sp):
    def variable_conductance(m, spread=3.0, seed=1):
        """Five-point stencil with random per-tile conductances — a die whose tiles differ.

        Built as an assembly over edges, so the result stays symmetric and diagonally dominant."""
        rng = np.random.default_rng(seed)
        kappa_tile = np.exp(rng.uniform(-spread, spread, size=(m, m)))
        rows, cols, vals = [], [], []
        diag = np.zeros(m * m)
        for i in range(m):
            for j in range(m):
                a = i * m + j
                for di, dj in ((0, 1), (1, 0)):
                    ii, jj = i + di, j + dj
                    if ii >= m or jj >= m:
                        continue
                    bnode = ii * m + jj
                    w = 2.0 / (1.0 / kappa_tile[i, j] + 1.0 / kappa_tile[ii, jj])  # harmonic
                    rows += [a, bnode]
                    cols += [bnode, a]
                    vals += [-w, -w]
                    diag[a] += w
                    diag[bnode] += w
        A = sp.coo_matrix((vals, (rows, cols)), shape=(m * m, m * m)).tocsr()
        A = A + sp.diags(diag + 0.4)
        return A.tocsr()
    return (variable_conductance,)


@app.cell
def _(A_grid, Ad_grid, a_norm_errors, conjugate_gradients, np, variable_conductance):
    def preconditioned_report(A, label):
        Ad = A.toarray()
        d = np.sqrt(A.diagonal())
        Aj = Ad / d[:, None] / d[None, :]                 # symmetric Jacobi preconditioning
        k_plain = np.linalg.cond(Ad)
        k_jac = np.linalg.cond(Aj)
        rng = np.random.default_rng(0)
        b = rng.standard_normal(A.shape[0])
        xs = conjugate_gradients(A, b, 300)
        e = a_norm_errors(A, xs, np.linalg.solve(Ad, b))
        bj = b / d
        Ajs = type(A)(Aj)
        xsj = conjugate_gradients(Ajs, bj, 300)
        ej = a_norm_errors(Ajs, xsj, np.linalg.solve(Aj, bj))
        hit = lambda v: int(np.argmax(v < 1e-10)) if (v < 1e-10).any() else None
        return dict(label=label, k_plain=k_plain, k_jac=k_jac,
                    it_plain=hit(e), it_jac=hit(ej))

    precond_rows = [
        preconditioned_report(A_grid, "constant conductance (our stencil)"),
        preconditioned_report(variable_conductance(24), "variable conductance"),
    ]
    _ = Ad_grid
    return (precond_rows,)


@app.cell
def _(mo, precond_rows):
    _rows = [
        f"| {r['label']} | {r['k_plain']:.1f} | {r['k_jac']:.1f} | {r['it_plain']} | {r['it_jac']} |"
        for r in precond_rows
    ]
    mo.md(
        "| matrix | κ(A) | κ(D^{-1/2}AD^{-1/2}) | CG iters | Jacobi-PCG iters |\n"
        "|:--|--:|--:|--:|--:|\n" + "\n".join(_rows) +
        """

The first row is the trap: the condition number does not move, and neither does the iteration
count. The second row is what preconditioning is supposed to look like.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. The probabilistic reading, and the handover to notebook 01

    Everything above is a statement about *optimality in a subspace*. The probabilistic-numerics reading changes the question from "which point in $\mathcal{K}_k$ is best?" to "what should I believe about $x_\ast$, having seen $k$ matrix–vector products?"

    Put a Gaussian prior on $x$, treat each product as a linear observation $s_i^\top A x_\ast = s_i^\top b$, and condition. **BayesCG** (Cockayne, Oates, Ipsen & Girolami 2019) shows that for a particular prior the posterior *mean* is exactly the CG iterate — so CG is the point estimate of a Bayesian procedure, and the posterior covariance is the uncertainty CG never reports.

    That is a satisfying result, and it inherits two properties that notebook 01 is a reaction to:

    * the belief is a **dense $n \times n$ covariance** — a global object, even though $A$ was sparse;
    * choosing the next $s_i$ is a **global policy**, and evaluating the belief needs the same inner products that made CG synchronise.

    Notebook 01 takes the other route: instead of a prior over $x$ and observations of $Ax$, it reads $A$ itself as a precision matrix, so that $p(x) = \mathcal{N}(A^{-1}b, A^{-1})$ and the sparsity of $A$ *is* a conditional independence structure. The belief then factorises over the graph, and the solver becomes message passing between neighbouring tiles: no inner products, no barriers, no dense covariance — at the cost of exactness on loopy graphs.

    Between them, these two notebooks are the trade the tutorial is about. CG is optimal in its subspace and pays for it in synchronisation; message passing is local and asynchronous and pays for it in accuracy of the second moment.

    ## Exercises

    1. **The termination property.** Run CG for exactly $n$ iterations on a small system ($n = 20$) in double precision. How close to zero is the final error, and how does that change if you scale the matrix to have $\kappa = 10^{10}$?
    2. **Loss of orthogonality.** Store all residuals and plot $|r_i^\top r_j|$ as a heatmap. In exact arithmetic it is diagonal; find where floating point breaks it, and compare that iteration with where the convergence curve flattens.
    3. **Clusters by hand.** Build a spectrum with two tight clusters plus one lonely outlying eigenvalue. Predict the iteration count before running it.
    4. **Counting barriers.** Instrument the CG loop to count inner products, and compare the total against the halo exchanges. Then do the same for Jacobi (notebook 01a) and for the GaBP solver in notebook 01.
    5. **Preconditioning as re-modelling.** Apply an incomplete-Cholesky preconditioner to the stencil and measure the iteration count. Then re-read notebook 01 §4.9 and explain the improvement in terms of *correlation length* rather than condition number.
    6. **The subspace claim.** Verify numerically that Jacobi's iterate after $k$ steps lies in $\mathcal{K}_k(A, b)$, and that its energy-norm error is never below CG's at the same $k$.

    ## References

    * Hestenes, M. R., & Stiefel, E. (1952). *Methods of conjugate gradients for solving linear systems*. J. Res. NBS, 49(6), 409–436.
    * Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. Chapters 6–9.
    * Greenbaum, A. (1997). *Iterative Methods for Solving Linear Systems*. SIAM. The definitive treatment of what finite precision does to CG.
    * Shewchuk, J. R. (1994). *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*. CMU technical report.
    * Cockayne, J., Oates, C. J., Ipsen, I. C. F., & Girolami, M. (2019). *A Bayesian conjugate gradient method*. Bayesian Analysis, 14(3), 937–1012.
    * Ghysels, P., & Vanroose, W. (2014). *Hiding global synchronization latency in the preconditioned conjugate gradient algorithm*. Parallel Computing, 40(7), 224–238.
    """
    )
    return


if __name__ == "__main__":
    app.run()
