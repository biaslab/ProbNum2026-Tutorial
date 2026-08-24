# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
"""Probabilistic Linear Algebra — Solving Ax = b as Inference.

ProbNum 2026 tutorial, notebook 1 of 5 (Python / Marimo edition).
"""
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return go, mo, np


@app.cell
def _():
    # Shared palette (matching the Julia/Pluto edition) and small plotting helpers.
    PAL = dict(blue="#2a78d6", black="#0b0b0b", green="#008300", orange="#eb6834",
               pink="#e87ba4", gray="#898781", violet="#4a3aa7", white="#fcfcfb")

    def base_layout(fig, title="", xlabel="", ylabel="", **kw):
        fig.update_layout(template="plotly_white", title=title,
                          xaxis_title=xlabel, yaxis_title=ylabel,
                          margin=dict(l=60, r=20, t=50, b=50), **kw)
        return fig

    def hex_rgba(hexcolor, alpha):
        h = hexcolor.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return PAL, base_layout, hex_rgba


@app.cell
def _(mo):
    mo.md(
        r"""
    # Probabilistic linear algebra: solving Ax = b as inference

    **ProbNum 2026 tutorial — notebook 1 of 5 (Python / Marimo edition)**

    Solving a linear system is the primitive underneath almost all of computational science — and it is also the cleanest place to see the central move of probabilistic numerics. An iterative solver does not get to *see* the solution $x_\ast = A^{-1}b$; it only gets to *query* the problem, one matrix–vector product at a time. Before any queries, the solver knows nothing about $x_\ast$; after $n$ independent queries, it knows everything. In between, it is in a state of **partial information** — and the honest description of partial information is a probability distribution.

    In this notebook we build a linear solver whose state is a Gaussian posterior over the solution: each matrix–vector product is an *observation*, conditioning is the *iteration*, and the classical conjugate gradient method falls out as the posterior mean under a particular prior. As in every notebook of this series, we finish by handing the whole construction to a message-passing formulation — here a small hand-rolled Gaussian belief propagation, the Python stand-in for the Julia edition's [RxInfer.jl](https://rxinfer.com).
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The tutorial series.** This is one of five notebooks, following the five classic problem settings of [probabilistic-numerics.org](https://www.probabilistic-numerics.org):

    1. **Linear algebra** — this notebook
    2. **Quadrature** — Bayesian quadrature
    3. **Optimization** — optimization as inference over the minimizer
    4. **Ordinary differential equations** — ODE filters and smoothers
    5. **Partial differential equations** — PDE solving as GP regression on the operator

    Each notebook exists in a Julia/Pluto edition and a Python/Marimo edition (this one).
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. The problem

    We want to solve

    $$
    A x = b, \qquad A \in \mathbb{R}^{n \times n} \text{ symmetric positive definite},
    $$

    in the regime where $A$ is too large or too expensive to factorize. Iterative methods (conjugate gradients, GMRES, …) only touch $A$ through **matrix–vector products**, and stop after $k \ll n$ of them, returning an approximation $x_k$. As with quadrature in notebook 2, the classical method returns a *number* (well, a vector) — and the classical error analysis bounds $\lVert x_k - x_\ast \rVert$ in terms of quantities like the condition number, which are exactly as unknown as the solution itself.

    The probabilistic-numerics reading (Hennig, 2015; Cockayne et al., 2019) goes like this. Choose a *search direction* $s_i \in \mathbb{R}^n$ and compute one matrix–vector product. Since $A x_\ast = b$, we then know the number

    $$
    y_i \;=\; s_i^\top A\, x_\ast \;=\; s_i^\top b .
    $$

    That is a **noise-free linear measurement of the unknown solution** — precisely the kind of data that Gaussian inference digests in closed form. A $k$-step iterative solver is an agent that has measured $k$ one-dimensional projections of $x_\ast$ and must report its belief about the rest.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Gaussian inference from matrix–vector products

    Put a Gaussian prior on the solution, $x \sim \mathcal{N}(\mu_0, \Sigma_0)$. Collect the $k$ search directions into $S = [s_1, \dots, s_k] \in \mathbb{R}^{n \times k}$ and write $M = S^\top A$ for the measurement operator. The data is $y = S^\top b = M x_\ast$. Conditioning a Gaussian on linear observations gives, in closed form,

    $$
    \begin{aligned}
    \mu_k &= \mu_0 + \Sigma_0 M^\top \left( M \Sigma_0 M^\top \right)^{-1} (y - M \mu_0), \\
    \Sigma_k &= \Sigma_0 - \Sigma_0 M^\top \left( M \Sigma_0 M^\top \right)^{-1} M \Sigma_0 .
    \end{aligned}
    $$

    (The same joint-Gaussian trick as in the quadrature notebook — there the linear functional of the latent object was an integral; here it is $k$ projections.) Three observations, worth making before any code:

    * **Each independent direction removes exactly one rank of uncertainty.** $\Sigma_k$ has rank $n - k$: the belief is *certain* along the measured directions $A^\top S$ and untouched orthogonal to them. Computation is information, in the literal sense.
    * **After $n$ independent directions the posterior collapses to a point mass at $x_\ast$.** A direct solver is just the fully-observed limit of an iterative one.
    * **For fixed directions, $\Sigma_k$ does not depend on $b$** — the uncertainty is determined by where you *looked*, not what you *saw*. (Adaptive policies below choose $s_i$ from the current residual, which reintroduces a dependence — that is what makes them good.)
    """
    )
    return


@app.cell
def _(np):
    def bayes_linsolve(A, b, S, mu0, Sigma0, eps=1e-6):
        "Posterior over x after observing the projections SᵀA x = Sᵀb."
        if S.shape[1] == 0:
            return mu0.copy(), Sigma0.copy()
        M = S.T @ A
        y = S.T @ b
        G = M @ Sigma0 @ M.T + eps * np.eye(M.shape[0])   # Gram matrix of the observations (+ jitter)
        C = Sigma0 @ M.T
        mu = mu0 + C @ np.linalg.solve(G, y - M @ mu0)
        Sigma = Sigma0 - C @ np.linalg.solve(G, C.T)
        return mu, 0.5 * (Sigma + Sigma.T)
    return (bayes_linsolve,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. The geometry, in two dimensions

    For $n = 2$ everything is visible. Each observation $s^\top A x = s^\top b$ defines a **hyperplane** (here: a line) that the solution must lie on. Conditioning intersects the prior with that hyperplane: the 2σ ellipse flattens onto the line after one observation and shrinks to (numerically) a point after two.

    Rotate the first search direction and watch *which* uncertainty survives. Any two independent directions identify the solution exactly — the choice of directions affects the *intermediate* beliefs, not the final one. Policies for choosing directions are the subject of the next section.
    """
    )
    return


@app.cell
def _(mo):
    theta = mo.ui.slider(0, 180, step=5, value=20, label="Angle of first search direction θ (deg)")
    k2d = mo.ui.slider(0, 2, step=1, value=1, label="Number of observations k")
    mo.vstack([theta, k2d])
    return k2d, theta


@app.cell
def _(PAL, base_layout, bayes_linsolve, go, k2d, np, theta):
    def _ellipse(mu, Sigma):
        w, V = np.linalg.eigh((Sigma + Sigma.T) / 2)
        t = np.linspace(0, 2 * np.pi, 200)
        pts = mu[:, None] + 2 * V @ (np.sqrt(np.maximum(w, 0.0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
        return pts[0], pts[1]

    _A2 = np.array([[3.0, 1.0], [1.0, 2.0]])
    _b2 = np.array([1.0, 2.0])
    _x2 = np.linalg.solve(_A2, _b2)
    _th = np.deg2rad(theta.value)
    _dirs = [np.array([np.cos(_th), np.sin(_th)]), np.array([-np.sin(_th), np.cos(_th)])]
    _S = np.zeros((2, 0)) if k2d.value == 0 else np.column_stack(_dirs[:k2d.value])
    _mu, _Sigma = bayes_linsolve(_A2, _b2, _S, np.zeros(2), np.eye(2), eps=1e-8)

    _fig = go.Figure()
    _px, _py = _ellipse(np.zeros(2), np.eye(2))
    _fig.add_trace(go.Scatter(x=_px, y=_py, mode="lines", name="prior (2σ)",
                              line=dict(color=PAL["gray"], dash="dash", width=1.5)))
    for _i, _s in enumerate(_dirs[:k2d.value]):
        _v = _A2.T @ _s               # observed hyperplane: v · x = s · b
        _cc = float(np.dot(_s, _b2))
        _xsl = np.linspace(-3, 3, 100)
        if abs(_v[1]) > 1e-8:
            _yl = (_cc - _v[0] * _xsl) / _v[1]
        else:
            _xsl = np.full(100, _cc / _v[0]); _yl = np.linspace(-2.5, 3, 100)
        _fig.add_trace(go.Scatter(x=_xsl, y=_yl, mode="lines",
                                  line=dict(color=PAL["violet"], dash="dot", width=1.5),
                                  name="observed hyperplanes" if _i == 0 else None,
                                  showlegend=_i == 0))
    _qx, _qy = _ellipse(_mu, _Sigma)
    _fig.add_trace(go.Scatter(x=_qx, y=_qy, mode="lines", name="posterior (2σ)",
                              line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=[_mu[0]], y=[_mu[1]], mode="markers", name="posterior mean",
                              marker=dict(color=PAL["blue"], size=9, line=dict(color=PAL["white"], width=1.5))))
    _fig.add_trace(go.Scatter(x=[_x2[0]], y=[_x2[1]], mode="markers", name="true solution",
                              marker=dict(color=PAL["black"], size=13, symbol="star")))
    base_layout(_fig, title=f"Belief about the solution (k = {k2d.value})", xlabel="x₁", ylabel="x₂",
                legend=dict(x=0.02, y=0.98))
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig.update_xaxes(range=[-3, 3]); _fig.update_yaxes(range=[-2.5, 3])
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Scaling up: policies for choosing directions

    Now a real (small) system: the tridiagonal matrix you get from discretizing the 1D *screened Poisson* equation $-u''(t) + c\, u(t) = f(t)$ on $[0,1]$, in the standard scaled form $A u = h^2 f$ with grid spacing $h$ — which keeps the entries of $A$ of order one, and the screening term keeps the system decently conditioned ($\kappa \approx 9$), so that every matrix–vector product carries visible information. The *solution of the linear system is a discretized function*, so we can draw the posterior over it as a curve with a credible band. (File this away for notebook 5, where PDE solving is the headline act.)

    The solver above accepts *any* directions. Where should an agent look? We compare three policies:

    * **residual** — measure along the current residual $r_i = b - A\mu_{i-1}$ (steepest descent, made Bayesian). Because the posterior mean reproduces all past observations exactly, each new residual is orthogonal to all previous directions: the policy automatically explores new territory.
    * **random** — measure along random directions. No knowledge of the problem, but $n$ of them are independent with probability 1.
    * **coordinate** — measure along $e_1, e_2, \dots$ in order. Reads off $A$'s rows one at a time.
    """
    )
    return


@app.cell
def _(np):
    n = 20
    _h = 1 / (n + 1)
    _c = 220.0   # screening / reaction coefficient
    _main = (2 + _c * _h**2) * np.ones(n)
    _off = -np.ones(n - 1)
    A = np.diag(_main) + np.diag(_off, 1) + np.diag(_off, -1)   # −u'' + c·u, scaled by h²
    ts = np.arange(1, n + 1) * _h                              # interior grid points
    b = 100 * _h**2 * np.exp(-(ts - 0.3)**2 / (2 * 0.1**2))    # localized bump forcing
    x_true = np.linalg.solve(A, b)
    Sigma0n = np.eye(n)                                        # prior: x ~ N(0, I)
    return A, Sigma0n, b, n, ts, x_true


@app.cell
def _(np):
    def conjugate_gradient(A, b, x0, k):
        "Textbook conjugate gradients, keeping iterates and search directions."
        x = x0.copy()
        r = b - A @ x
        p = r.copy()
        iterates = [x.copy()]
        directions = []
        for _ in range(k):
            Ap = A @ p
            alpha = (r @ r) / (p @ Ap)
            directions.append(p.copy())
            x = x + alpha * p
            rnew = r - alpha * Ap
            beta = (rnew @ rnew) / (r @ r)
            p = rnew + beta * p
            r = rnew
            iterates.append(x.copy())
        return dict(iterates=iterates, directions=directions)
    return (conjugate_gradient,)


@app.cell
def _(bayes_linsolve, np):
    def sequential_solve(A, b, policy, k, mu0, Sigma0, eps=1e-6, rng=None):
        rng = np.random.default_rng(2026) if rng is None else rng
        nn = len(b)
        S = np.zeros((nn, 0))
        hist = [dict(mu=mu0.copy(), Sigma=Sigma0.copy())]
        mu = mu0.copy()
        for i in range(k):
            if policy == "random":
                s = rng.standard_normal(nn)
            elif policy == "coordinate":
                s = np.eye(nn)[:, i % nn]
            else:                       # "residual" policy
                s = b - A @ mu
            if np.linalg.norm(s) < 1e-10 * np.linalg.norm(b):   # already converged
                hist.append(hist[-1])
                continue
            S = np.column_stack([S, s / np.linalg.norm(s)])
            mu, Sigma = bayes_linsolve(A, b, S, mu0, Sigma0, eps)
            hist.append(dict(mu=mu, Sigma=Sigma))
        return dict(hist=hist, S=S)
    return (sequential_solve,)


@app.cell
def _(A, Sigma0n, b, n, np, sequential_solve):
    runs = {p: sequential_solve(A, b, p, n, np.zeros(n), Sigma0n)
            for p in ["residual", "random", "coordinate"]}
    return (runs,)


@app.cell
def _(mo):
    policy = mo.ui.dropdown(
        options={"residual (steepest-descent directions)": "residual",
                 "random directions": "random",
                 "coordinate directions": "coordinate"},
        value="residual (steepest-descent directions)", label="Direction policy")
    k_iter = mo.ui.slider(0, 20, step=1, value=6, label="Number of matrix–vector products k")
    mo.vstack([policy, k_iter])
    return k_iter, policy


@app.cell
def _(PAL, base_layout, go, hex_rgba, k_iter, np, policy, runs, ts, x_true):
    _state = runs[policy.value]["hist"][k_iter.value]
    _sd = np.sqrt(np.maximum(np.diag(_state["Sigma"]), 0.0))
    _up, _lo = _state["mu"] + 2 * _sd, _state["mu"] - 2 * _sd

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([ts, ts[::-1]]),
                              y=np.concatenate([_up, _lo[::-1]]), fill="toself",
                              fillcolor=hex_rgba(PAL["blue"], 0.15),
                              line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                              name="± 2σ", showlegend=True))
    _fig.add_trace(go.Scatter(x=ts, y=_state["mu"], mode="lines+markers",
                              line=dict(color=PAL["blue"], width=2),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1)),
                              name="posterior mean"))
    _fig.add_trace(go.Scatter(x=ts, y=x_true, mode="lines", name="true solution",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    _n = len(x_true)
    base_layout(_fig, title=f"Belief about the solution ({policy.value}, k = {k_iter.value} of {_n})",
                xlabel="t", ylabel="u(t)", legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig
    return


@app.cell
def _(k_iter, mo, np, policy, runs, x_true):
    _st = runs[policy.value]["hist"][k_iter.value]
    _err = np.linalg.norm(_st["mu"] - x_true)
    _unc = np.sqrt(np.trace(_st["Sigma"]))
    _rank = np.linalg.matrix_rank(_st["Sigma"], tol=1e-6)
    mo.md(
        f"""
    | quantity | value |
    |:---|---:|
    | error ‖μₖ − x*‖₂ | {_err:.3g} |
    | posterior uncertainty √tr(Σₖ) | {_unc:.3g} |
    | rank of Σₖ (numerical) | {_rank} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for**

    * With the **residual** policy the band collapses first where the forcing pushes the solution around — the agent spends its budget where the residual says the belief is worst.
    * With **coordinate** directions the belief becomes certain about the first $k$ components and stays maximally uncertain about the rest: you can see the untouched prior band on the right side of the plot.
    * The numerical rank of $\Sigma_k$ drops by exactly one per (independent) observation, policy be damned — the *quality* of the posterior mean differs enormously, the *quantity* of information does not.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Convergence — and conjugate gradients as a posterior mean

    How fast does the posterior mean approach the truth, per matrix–vector product? And does the posterior's *own* uncertainty estimate track the actual error? Below we also run classical **conjugate gradients** from the same starting point as the reference method. (The plot reacts to nothing — it always runs all policies to $k = n$.)
    """
    )
    return


@app.cell
def _(A, PAL, base_layout, b, conjugate_gradient, go, n, np, runs, x_true):
    _ks = np.arange(0, n + 1)
    _cg = conjugate_gradient(A, b, np.zeros(n), n)
    _fig = go.Figure()
    for _p, _col in [("residual", PAL["blue"]), ("random", PAL["green"]), ("coordinate", PAL["pink"])]:
        _errs = [np.linalg.norm(h["mu"] - x_true) for h in runs[_p]["hist"]]
        _fig.add_trace(go.Scatter(x=_ks, y=np.maximum(_errs, 1e-16), mode="lines",
                                  line=dict(color=_col, width=2), name=f"posterior mean — {_p}"))
    _std = [np.sqrt(np.trace(h["Sigma"])) for h in runs["residual"]["hist"]]
    _fig.add_trace(go.Scatter(x=_ks, y=np.maximum(_std, 1e-16), mode="lines",
                              line=dict(color=PAL["blue"], width=2, dash="dash"),
                              name="posterior std √tr(Σₖ) — residual"))
    _cgerr = [np.linalg.norm(xk - x_true) for xk in _cg["iterates"]]
    _fig.add_trace(go.Scatter(x=_ks, y=np.maximum(_cgerr, 1e-16), mode="lines+markers",
                              line=dict(color=PAL["orange"], width=2, dash="dashdot"),
                              marker=dict(size=6, symbol="diamond", line=dict(color=PAL["white"], width=1)),
                              name="conjugate gradients (classical)"))
    base_layout(_fig, title="Convergence of the posterior mean",
                xlabel="matrix–vector products k", ylabel="‖μₖ − x*‖₂",
                legend=dict(x=0.02, y=0.02))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The residual policy rides close to CG; random and coordinate directions pay for their ignorance of the problem. (The error floor around $10^{-7}$ is the $\varepsilon = 10^{-6}$ jitter, same story as in the quadrature notebook.)

    The dashed curve deserves a hard look: with the isotropic prior $\Sigma_0 = \tau^2 I$, the posterior std declines on an essentially *fixed schedule* — $\sqrt{\operatorname{tr}(\Sigma_k)} = \tau\sqrt{n - k}$ for orthonormal directions — because the solver is counting **dimensions explored**, not error magnitude. That is honest (the true error is indeed far below it) but uselessly conservative. Calibrated uncertainty requires a prior that knows the *scale* of the problem — which is exactly where the following classical connection comes from.

    **The classical method is a point estimate of the Bayesian one.** Make the prior scale-aware, $\Sigma_0 = A^{-1}$ (unavailable in practice, but crisp in theory), and condition on the directions that CG itself generates. The posterior mean then minimizes the $A$-norm error over the Krylov space spanned by those directions — which is the *defining property* of the CG iterate. So the $k$-th CG iterate **is** the posterior mean of this Gaussian solver (Hennig, 2015; Cockayne et al., 2019). Numerically, after $k = 8$ steps:
    """
    )
    return


@app.cell
def _(A, b, bayes_linsolve, conjugate_gradient, n, np):
    _k = 8
    _cg = conjugate_gradient(A, b, np.zeros(n), _k)
    _S = np.column_stack(_cg["directions"])
    _mu, _ = bayes_linsolve(A, b, _S, np.zeros(n), np.linalg.inv(A), eps=0.0)
    cg_vs_bayes = float(np.max(np.abs(_mu - _cg["iterates"][-1])))
    return (cg_vs_bayes,)


@app.cell
def _(cg_vs_bayes, mo):
    mo.callout(
        mo.md(
            rf"""
    **Punchline.** Max componentwise difference between the CG iterate and the posterior mean under the $\Sigma_0 = A^{{-1}}$ prior: **{cg_vs_bayes:.3g}** — they are the same algorithm. CG has been maintaining a Gaussian posterior since 1952; it just never told anyone the covariance.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. The same solver as message passing

    The bespoke conditioning formulas of Section 2 are exactly what sum-product message passing derives automatically for a linear-Gaussian factor graph. The model is two lines: a Gaussian prior over the latent solution, observed through the constant matrix $M = S^\top A$:

    ```
        N(μ₀, Σ₀) ──▶ (x) ──▶ [ SᵀA · ] ──▶ N(·, εI) ──▶ (y = Sᵀb, observed)
    ```

    The graph is a tree and everything is conjugate, so belief propagation is exact and we can check it digit-for-digit. The Julia edition runs this on [RxInfer.jl](https://rxinfer.com); here we hand-roll the two Gaussian messages in NumPy — the *prior factor* and the *likelihood factor* — and take their product in information (canonical) form. (We take the directions from the policy and $k$ selected above, with at least one observation.)
    """
    )
    return


@app.cell
def _(np):
    def gaussian_bp(m0, S0, M, R, y):
        """Sum-product update for a linear-Gaussian tree, in information (canonical) form:
        the prior factor and the likelihood factor y = M x + N(0, R) meet at x and multiply."""
        Lam0 = np.linalg.inv(S0)              # prior factor  → precision, natural mean
        eta0 = Lam0 @ m0
        Rinv = np.linalg.inv(R)
        Lam_lik = M.T @ Rinv @ M              # message from the likelihood factor to x
        eta_lik = M.T @ Rinv @ y
        Lam = Lam0 + Lam_lik                  # product of the two Gaussian messages
        S = np.linalg.inv(Lam)
        return S @ (eta0 + eta_lik), 0.5 * (S + S.T)
    return (gaussian_bp,)


@app.cell
def _(A, b, k_iter, np, policy, runs):
    _k_rx = max(k_iter.value, 1)
    _run = runs[policy.value]
    _Sk = _run["S"][:, :min(_k_rx, _run["S"].shape[1])]
    _kk = _Sk.shape[1]
    rx_M = _Sk.T @ A
    rx_y = _Sk.T @ b
    rx_R = 1e-6 * np.eye(_kk)
    rx_cf = _run["hist"][_k_rx]     # closed-form posterior after k_rx observations
    return rx_M, rx_R, rx_cf, rx_y


@app.cell
def _(Sigma0n, gaussian_bp, n, np, rx_M, rx_R, rx_y):
    rx_mu, rx_Sigma = gaussian_bp(np.zeros(n), Sigma0n, rx_M, rx_R, rx_y)
    return rx_Sigma, rx_mu


@app.cell
def _(mo, np, rx_Sigma, rx_cf, rx_mu):
    _dmu = np.max(np.abs(rx_mu - rx_cf["mu"]))
    _dSig = np.max(np.abs(rx_Sigma - rx_cf["Sigma"]))
    mo.md(
        f"""
    | quantity | max abs. difference (closed form vs message passing) |
    |:---|---:|
    | posterior mean μₖ | {_dmu:.3g} |
    | posterior covariance Σₖ | {_dSig:.3g} |
    """
    )
    return


@app.cell
def _(PAL, base_layout, go, rx_cf, rx_mu, ts, x_true):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=ts, y=rx_cf["mu"], mode="lines", name="closed form",
                              line=dict(color=PAL["blue"], width=6), opacity=0.35))
    _fig.add_trace(go.Scatter(x=ts, y=rx_mu, mode="lines", name="message passing",
                              line=dict(color=PAL["green"], width=2, dash="dash")))
    _fig.add_trace(go.Scatter(x=ts, y=x_true, mode="lines", name="true solution",
                              line=dict(color=PAL["black"], width=1.5, dash="dot")))
    base_layout(_fig, title="Same posterior, two routes", xlabel="t", ylabel="u(t)",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Where to go from here

    * **Priors are preconditioners.** The choice of $\Sigma_0$ plays exactly the role a preconditioner plays classically: it encodes what you know about the geometry of $A$ before you start. $\Sigma_0 = A^{-1}$ recovers CG; cheap approximations of it (diagonal, incomplete factorizations) give *probabilistic preconditioned* solvers.
    * **Matrix-based inference.** We inferred the solution $x$ for one right-hand side. The richer view (Hennig, 2015; Wenger & Hennig, 2020) places the prior on $A^{-1}$ itself, so that information accumulated for one $b$ transfers to the next — a solver that *learns the matrix*.
    * **Calibration is the hard part.** With $k < n$ observations, the posterior in the unexplored subspace is pure prior; whether the credible band is honest depends entirely on $\Sigma_0$. See Cockayne et al. (2019) and Bartels et al. (2019) for the state of the debate.
    * **The pattern to remember** — latent quantity, linear observations, Gaussian messages — is the same one that runs quadrature in notebook 2, and it is the same one that will integrate ODEs in notebook 4. Only the meaning of the latent vector changes.

    ### Exercises

    1. **Geometry.** In the 2D demo, find the angle $\theta$ for which a *single* observation already places the posterior mean (almost) on the true solution. What is special about that direction?
    2. **Calibration.** Fix the residual policy and $k = 6$, and sweep the prior scale $\tau$ in $\Sigma_0 = \tau^2 I$ from 0.5 to 50. Plot the ratio of actual error to posterior std. Where is it calibrated, overconfident, underconfident?
    3. **Priors as preconditioners.** Re-run the convergence experiment with $\Sigma_0 = \operatorname{diag}(A)^{-1}$ and with $\Sigma_0 = A^{-1}$. Explain the ordering of the three curves.
    4. **Transfer.** Solve a *second* system $A x = b'$ (new forcing) reusing the directions $S$ collected for $b$. Compare random vs residual directions. Why does one policy's information transfer and the other's barely does?
    5. **Streaming inference.** Rewrite the message-passing section to process observations *one at a time*, feeding each posterior back as the next prior. Verify the final posterior equals the batch one. (This is the gateway drug to the filtering view of notebook 4.)

    ### References

    * Hennig, P. (2015). *Probabilistic interpretation of linear solvers*. SIAM Journal on Optimization, 25(1), 234–260.
    * Cockayne, J., Oates, C. J., Ipsen, I. C. F., & Girolami, M. (2019). *A Bayesian conjugate gradient method*. Bayesian Analysis, 14(3), 937–1012.
    * Bartels, S., Cockayne, J., Ipsen, I. C. F., & Hennig, P. (2019). *Probabilistic linear solvers: a unifying view*. Statistics and Computing, 29, 1249–1263.
    * Wenger, J., & Hennig, P. (2020). *Probabilistic linear solvers for machine learning*. NeurIPS.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press. Free PDF at [probabilistic-numerics.org](https://www.probabilistic-numerics.org).
    """
    )
    return


if __name__ == "__main__":
    app.run()
