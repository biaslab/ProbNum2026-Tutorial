# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
"""Probabilistic PDE Solvers — Boundary Value Problems as GP Regression.

ProbNum 2026 tutorial, notebook 5 of 5 (Python / Marimo edition).
"""
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np


@app.cell
def _():
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
    # Probabilistic PDE solvers: boundary value problems as GP regression

    **ProbNum 2026 tutorial — notebook 5 of 5 (Python / Marimo edition)**

    A partial differential equation is a statement about a function we cannot see: it constrains the solution $u$ through a *differential operator*, and a numerical PDE solver must reconstruct $u$ from finitely many evaluations of that constraint. Finite differences, finite elements, spectral collocation — each is a rule for turning "the operator equals $f$ here, and the boundary equals $g$ there" into a solution estimate.

    In this notebook we rebuild an elliptic PDE solver as **Gaussian-process regression on the operator**. Put a GP prior over the solution $u$; because a linear differential operator $\mathcal{L}$ is a *linear map*, $\mathcal{L}u$ is again a Gaussian process, and each interior collocation condition $\mathcal{L}u(x)=f(x)$ — with each boundary condition — is a linear-Gaussian observation of it. Conditioning gives a posterior over the whole solution in one step (Cockayne et al., 2017). And the punchline returns: the **finite-difference solution drops out as a posterior mean**, with the kernel supplying the credible band the classical solver never kept. As always, we close with a message-passing formulation (the Julia edition's [RxInfer.jl](https://rxinfer.com); here a hand-rolled Gaussian belief propagation).
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The tutorial series.** One of five notebooks following [probabilistic-numerics.org](https://www.probabilistic-numerics.org): (1) linear algebra, (2) quadrature, (3) optimization, (4) ODE solvers, (5) **PDE solvers — this notebook**.
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

    We want the solution of a linear **boundary value problem** — the 1-D Poisson equation, the simplest elliptic PDE:

    $$
    -\,u''(x) \;=\; f(x), \qquad x \in (0, 1), \qquad u(0) = a, \quad u(1) = b .
    $$

    A classical solver discretizes the domain, replaces $-u''$ by a finite-difference stencil, and solves the resulting linear system. Unlike the ODE of notebook 4, there is **no direction to march in**: the condition at $x=1$ must influence the estimate at $x=0$. A boundary value problem is a *global* inference, not a forward recursion.

    The probabilistic reading: the solution $u(\cdot)$ is a **latent function**, known only through the operator $\mathcal{L} = -\tfrac{d^2}{dx^2}$ at interior points and through the two boundary values. Our running example is a manufactured problem with a known closed-form solution, for grading:

    $$
    u_\star(x) = 1 + 2x + \sin(\pi x) + \tfrac{1}{2}\sin(3\pi x),
    \qquad f(x) = \pi^2 \sin(\pi x) + \tfrac{9}{2}\pi^2 \sin(3\pi x),
    $$

    with $a = u_\star(0) = 1$ and $b = u_\star(1) = 3$. The linear term $1 + 2x$ lives in the null space of $\mathcal{L}$, so it is pinned down *only* by the boundary conditions — a test of whether the solver propagates information across the whole domain.
    """
    )
    return


@app.cell
def _(np):
    def utrue(x):
        return 1 + 2 * x + np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    def frhs(x):
        return np.pi**2 * np.sin(np.pi * x) + 4.5 * np.pi**2 * np.sin(3 * np.pi * x)   # = −u''
    a_bc, b_bc = float(utrue(0.0)), float(utrue(1.0))
    return a_bc, b_bc, frhs, utrue


@app.cell
def _(PAL, a_bc, b_bc, base_layout, frhs, go, make_subplots, np, utrue):
    _xs = np.linspace(0, 1, 300)
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    _fig.add_trace(go.Scatter(x=_xs, y=utrue(_xs), mode="lines", name="true solution u⋆ (unknown)",
                              line=dict(color=PAL["black"], width=2.5)), row=1, col=1)
    _fig.add_trace(go.Scatter(x=[0, 1], y=[a_bc, b_bc], mode="markers", name="boundary data u(0)=a, u(1)=b",
                              marker=dict(color=PAL["blue"], size=10, line=dict(color=PAL["white"], width=1.5))), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_xs, y=frhs(_xs), mode="lines", name="source f = −u''",
                              line=dict(color=PAL["orange"], width=2)), row=2, col=1)
    _fig.update_yaxes(title_text="u(x)", row=1, col=1); _fig.update_yaxes(title_text="f(x)", row=2, col=1)
    _fig.update_xaxes(title_text="x", row=2, col=1)
    base_layout(_fig, title="The solver never sees u — only the operator and the boundary")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. The prior: a Gaussian process over the solution

    We place a GP prior over the solution, $u \sim \mathcal{GP}(0, k)$, with a squared-exponential kernel $k(x, x') = \exp(-(x-x')^2 / 2\ell^2)$ whose lengthscale $\ell$ states how smooth we believe the solution is. The one fact that makes this a *PDE* solver rather than an interpolator is linearity: **applying a linear operator to a GP yields another GP**, so the solution $u$, its second derivative, the source $f$, and the boundary values all live in **one jointly Gaussian family**. In the discretization below we realise $\mathcal{L}$ as a finite-difference stencil, so the operator becomes a matrix $A$ acting on the grid values — but the picture is exactly the one above.

    Samples from the prior — before any operator or boundary condition is imposed. Each collocation condition in the next section slices this cloud down to the functions that obey the PDE.
    """
    )
    return


@app.cell
def _(np):
    def sekernel(x, xp, l):
        "Squared-exponential kernel — a smoothness prior over the solution function."
        return np.exp(-(x - xp)**2 / (2 * l**2))

    def assemble_pde(prob, n_nodes, n_colloc, l=0.2, s2=1e-8):
        """Assemble the discretized GP-collocation problem: grid X, GP prior covariance K,
        operator matrix A (two boundary rows + interior discrete-Laplacian rows), data c, jitter R.
        Fewer collocation points than interior nodes → a genuine meshless posterior."""
        utrue, frhs, a_bc, b_bc = prob
        X = np.linspace(0.0, 1.0, n_nodes)
        h = X[1] - X[0]
        K = sekernel(X[:, None], X[None, :], l) + 1e-10 * np.eye(n_nodes)
        idx = np.unique(np.round(np.linspace(1, n_nodes - 2, n_colloc)).astype(int))
        rows, c = [], []
        def e(i):
            r = np.zeros(n_nodes); r[i] = 1.0; return r
        rows.append(e(0)); c.append(a_bc)                     # boundary u(0) = a
        rows.append(e(n_nodes - 1)); c.append(b_bc)           # boundary u(1) = b
        for j in idx:                                          # interior collocation L u = −u'' = f
            r = np.zeros(n_nodes)
            r[j - 1] = -1 / h**2; r[j] = 2 / h**2; r[j + 1] = -1 / h**2
            rows.append(r); c.append(frhs(X[j]))
        A = np.array(rows); c = np.array(c)
        R = s2 * np.diag(np.diag(A @ K @ A.T) + 1e-12)
        return dict(X=X, h=h, K=K, A=A, c=c, R=R, colloc=X[idx])
    return assemble_pde, sekernel


@app.cell
def _(mo):
    ell_prior = mo.ui.slider(0.05, 0.6, step=0.05, value=0.2, label="Lengthscale ℓ")
    ell_prior
    return (ell_prior,)


@app.cell
def _(PAL, base_layout, ell_prior, go, hex_rgba, np, sekernel):
    _rng = np.random.default_rng(2026)
    _X = np.linspace(0, 1, 200)
    _K = sekernel(_X[:, None], _X[None, :], ell_prior.value) + 1e-9 * np.eye(200)
    _L = np.linalg.cholesky(_K)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([_X, _X[::-1]]),
                              y=np.concatenate([2 * np.ones(200), -2 * np.ones(200)]), fill="toself",
                              fillcolor=hex_rgba(PAL["blue"], 0.12), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="prior mean ± 2σ"))
    for _s in range(8):
        _fig.add_trace(go.Scatter(x=_X, y=_L @ _rng.standard_normal(200), mode="lines",
                                  line=dict(color=PAL["blue"], width=1.1), opacity=0.55,
                                  showlegend=_s == 0, name="prior samples", hoverinfo="skip"))
    base_layout(_fig, title=f"Prior samples from the GP over the solution (ℓ = {ell_prior.value})",
                xlabel="x", ylabel="u(x)", legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(range=[-3.2, 3.2])
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. The operator as observations

    Discretize the domain into $n$ grid points and stack every linear constraint into a single observation model $A\,u = c$:

    * two **boundary rows** — $u(0) = a$ and $u(1) = b$ — selecting the endpoints of $u$;
    * one **interior row** per collocation point $x_j$, the discrete Laplacian $\dfrac{2u_j - u_{j-1} - u_{j+1}}{h^2} = f(x_j)$ — the operator $\mathcal{L}u = -u''$ evaluated at $x_j$.

    Every row is a linear functional of the latent $u$, so with the GP prior $u \sim \mathcal{N}(0, K)$ and a small slack $R$, the posterior is one **Gaussian conditioning**:

    $$
    m = K A^\top \big(A K A^\top + R\big)^{-1} c, \qquad
    P = K - K A^\top \big(A K A^\top + R\big)^{-1} A K .
    $$

    The mean $m$ is the reconstructed solution; the diagonal of $P$ gives a pointwise $\pm 2\sigma$ band — widest where collocation points are sparse. There is nothing sequential here: boundary conditions and all interior residuals are fused at once, which is what an elliptic problem demands.
    """
    )
    return


@app.cell
def _(assemble_pde, np):
    def pde_gp_solve(prob, n_nodes, n_colloc, l=0.2, s2=1e-8):
        "Solve −u''=f, u(0)=a, u(1)=b by GP regression on the operator (one Gaussian conditioning)."
        d = assemble_pde(prob, n_nodes, n_colloc, l, s2)
        K, A, c, R = d["K"], d["A"], d["c"], d["R"]
        M = A @ K @ A.T + R
        G = K @ A.T @ np.linalg.inv(M)                  # gain: K Aᵀ (A K Aᵀ + R)⁻¹
        m = G @ c
        P = K - G @ (A @ K)
        return dict(X=d["X"], mean=m, std=np.sqrt(np.maximum(np.diag(P), 0.0)), cov=P,
                    A=A, K=K, c=c, R=R, colloc=d["colloc"], n_colloc=len(d["colloc"]))
    return (pde_gp_solve,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Try it

    The first panel shows the posterior mean and its $\pm 2\sigma$ band against the truth, with collocation points marked; the second is the pointwise error.

    Things to try:
    * Start with few collocation points and increase them: the band collapses onto the truth, fastest near the collocation sites. With too few points the solver is honestly uncertain *between* them.
    * Shrink the lengthscale $\ell$: a rougher prior fits sharp features but widens the bands; an over-smooth prior biases the mean where the true solution oscillates.
    * Watch the endpoints and the $1 + 2x$ trend: with only two boundary observations the solver still recovers the linear part, because the conditioning is global.
    """
    )
    return


@app.cell
def _(mo):
    n_nodes = mo.ui.slider(11, 81, step=2, value=41, label="Grid resolution n (nodes)")
    n_colloc = mo.ui.slider(3, 40, step=1, value=10, label="Interior collocation points")
    ell_solve = mo.ui.slider(0.05, 0.6, step=0.05, value=0.2, label="Lengthscale ℓ")
    log_s2 = mo.ui.slider(-12, -2, step=1, value=-8, label="Slack log₁₀σ²")
    mo.vstack([n_nodes, n_colloc, ell_solve, log_s2])
    return ell_solve, log_s2, n_colloc, n_nodes


@app.cell
def _(a_bc, b_bc, ell_solve, frhs, log_s2, n_colloc, n_nodes, pde_gp_solve, utrue):
    prob = (utrue, frhs, a_bc, b_bc)
    sol = pde_gp_solve(prob, n_nodes.value, min(n_colloc.value, n_nodes.value - 2),
                       l=ell_solve.value, s2=10.0**log_s2.value)
    return prob, sol


@app.cell
def _(PAL, base_layout, ell_solve, go, hex_rgba, make_subplots, n_nodes, np, sol, utrue):
    _up, _lo = sol["mean"] + 2 * sol["std"], sol["mean"] - 2 * sol["std"]
    _err = np.abs(sol["mean"] - utrue(sol["X"]))
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09)
    _fig.add_trace(go.Scatter(x=np.concatenate([sol["X"], sol["X"][::-1]]), y=np.concatenate([_up, _lo[::-1]]),
                              fill="toself", fillcolor=hex_rgba(PAL["blue"], 0.18), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="± 2σ"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=sol["X"], y=sol["mean"], mode="lines", name="posterior mean",
                              line=dict(color=PAL["blue"], width=2)), row=1, col=1)
    _fig.add_trace(go.Scatter(x=np.linspace(0, 1, 300), y=utrue(np.linspace(0, 1, 300)), mode="lines",
                              name="true solution", line=dict(color=PAL["black"], width=1.5, dash="dash")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=sol["colloc"], y=utrue(sol["colloc"]), mode="markers", name="collocation points",
                              marker=dict(color=PAL["orange"], size=7, line=dict(color=PAL["white"], width=1))), row=1, col=1)
    _fig.add_trace(go.Scatter(x=sol["X"], y=_err, mode="lines", name="pointwise error",
                              line=dict(color=PAL["green"], width=2)), row=2, col=1)
    _fig.add_trace(go.Scatter(x=sol["X"], y=2 * sol["std"], mode="lines", name="2σ band",
                              line=dict(color=PAL["blue"], width=1.5, dash="dot")), row=2, col=1)
    _fig.update_yaxes(title_text="u(x)", row=1, col=1); _fig.update_yaxes(title_text="|error|", row=2, col=1)
    _fig.update_xaxes(title_text="x", row=2, col=1)
    base_layout(_fig, title=f"Belief about the solution (n = {n_nodes.value}, {sol['n_colloc']} colloc, ℓ = {ell_solve.value})")
    _fig
    return


@app.cell
def _(mo, np, sol, utrue):
    _err = sol["mean"] - utrue(sol["X"])
    _cov = np.mean(np.abs(_err) <= 2 * sol["std"] + 1e-12)
    mo.md(
        f"""
    | quantity | value |
    |:---|---:|
    | max abs. error of the posterior mean | {np.max(np.abs(_err)):.3g} |
    | RMS error over the grid | {np.sqrt(np.mean(_err**2)):.3g} |
    | max 2σ band width | {2 * np.max(sol['std']):.3g} |
    | fraction of grid inside the ± 2σ band | {_cov:.3g} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for**

    * The band is not decoration: where collocation points are sparse the $2\sigma$ band swells, and the true error stays (mostly) inside it.
    * Push the collocation points up to $n - 2$ (all interior nodes): the band nearly vanishes and the mean becomes the finite-difference solution — the subject of the next section.
    * The coverage fraction is the calibration check. Under a badly chosen lengthscale it drops below 1, the solver's way of admitting the prior was wrong.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Convergence, calibration — and the finite-difference method as a posterior mean

    With **all** interior nodes used as collocation points the system $A u = c$ is square, and the solver reduces to the classical finite-difference scheme — so its convergence should be the textbook $\mathcal{O}(h^2)$. The dashed curve tracks a genuinely *meshless* run (half as many collocation points as nodes) to show the credible band shrinking in step.
    """
    )
    return


@app.cell
def _(np, pde_gp_solve, prob, utrue):
    def _conv():
        ns = [8, 16, 32, 64, 128]
        full, band = [], []
        for n in ns:
            s = pde_gp_solve(prob, n + 1, n - 1, l=0.2, s2=1e-12)     # collocation at every interior node
            full.append(np.max(np.abs(s["mean"] - utrue(s["X"]))))
            sm = pde_gp_solve(prob, n + 1, max((n + 1) // 2, 2), l=0.2, s2=1e-8)
            band.append(2 * np.max(sm["std"]))
        return np.array(ns), np.array(full), np.array(band)
    conv_ns, conv_full, conv_band = _conv()
    return conv_band, conv_full, conv_ns


@app.cell
def _(PAL, base_layout, conv_band, conv_full, conv_ns, go, np):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=conv_ns, y=np.maximum(conv_full, 1e-16), mode="lines+markers",
                              line=dict(color=PAL["blue"], width=2), marker=dict(size=7, line=dict(color=PAL["white"], width=1)),
                              name="posterior mean error (full collocation)"))
    _fig.add_trace(go.Scatter(x=conv_ns, y=conv_full[1] * (conv_ns[1] / conv_ns)**2, mode="lines",
                              line=dict(color=PAL["black"], width=1, dash="dot"), name="O(h²) reference"))
    _fig.add_trace(go.Scatter(x=conv_ns, y=conv_band, mode="lines+markers",
                              line=dict(color=PAL["pink"], width=2, dash="dash"),
                              marker=dict(size=7, symbol="diamond", line=dict(color=PAL["white"], width=1)),
                              name="meshless 2σ band (½ as many colloc)"))
    base_layout(_fig, title="Convergence of the posterior mean",
                xlabel="grid intervals n (= h⁻¹)", ylabel="error / band width", legend=dict(x=0.02, y=0.02))
    _fig.update_xaxes(type="log"); _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The mean converges at the dotted $\mathcal{O}(h^2)$ slope — the second-order rate of the central-difference stencil — and the meshless band contracts alongside it.

    **The classical method inside.** Take the square case: collocation at every interior node, and vanishing slack $\sigma^2 \to 0$. Then $A$ is square and invertible, and the posterior mean is $m = K A^\top (A K A^\top)^{-1} c = A^{-1} c$, **independent of the kernel $K$ entirely**. But $A u = c$ is precisely the finite-difference linear system, so $m = A^{-1}c$ is the **classical finite-difference solution**, to the last digit. The prior washes out of the mean the moment the data determine $u$ uniquely; what it leaves behind is the covariance, the calibrated uncertainty the finite-difference solver never computed. Numerically:
    """
    )
    return


@app.cell
def _(a_bc, b_bc, frhs, np, pde_gp_solve, prob, utrue):
    def _fd():
        n = 32; X = np.linspace(0, 1, n + 1); h = 1 / n; Ni = n - 1
        T = (np.diag(np.full(Ni, 2 / h**2)) + np.diag(np.full(Ni - 1, -1 / h**2), 1)
             + np.diag(np.full(Ni - 1, -1 / h**2), -1))
        rhs = np.array([frhs(X[j + 1]) for j in range(Ni)])
        rhs[0] += a_bc / h**2; rhs[-1] += b_bc / h**2         # move known boundary terms to the RHS
        u_fd = np.concatenate([[a_bc], np.linalg.solve(T, rhs), [b_bc]])   # classical finite-difference solution
        s = pde_gp_solve(prob, n + 1, n - 1, l=0.2, s2=1e-12)
        return np.max(np.abs(s["mean"] - u_fd)), np.max(np.abs(s["mean"] - utrue(X)))
    fd_vs_fd, fd_vs_true = _fd()
    return fd_vs_fd, fd_vs_true


@app.cell
def _(fd_vs_fd, fd_vs_true, mo):
    mo.callout(
        mo.md(
            rf"""
    **Punchline.** Posterior mean vs finite-difference solution: **{fd_vs_fd:.3g}**; posterior mean vs true solution: **{fd_vs_true:.3g}**. The finite-difference method *is* the GP-collocation posterior mean in the square, zero-slack limit — and the second, larger number is the $\mathcal{{O}}(h^2)$ discretization error the classical solver commits silently but the posterior covariance quantifies. The classical solver was Bayesian all along; it just discarded the covariance (Cockayne et al., 2019).
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

    Section 3's conditioning is a single linear-Gaussian model — a latent function-value vector $u$ with a GP prior, observed through the operator matrix $A$:

    ```
        N(0, K) ──▶ (u) ──▶ [ A· ] ── N(·, R) ──▶ (y = c, observed)
    ```

    — the direct sibling of notebook 1's $Ax=b$-as-inference, now with a *function-space* prior $K$ and the differential operator in the observation. The whole PDE is one conjugate Gaussian model (no chain — the elliptic problem is global), so a single message-passing update does the entire solve. Because the kernel prior $K$ is badly ill-conditioned, we schedule the sum-product messages in **moment form** (inverting only the well-conditioned observation-space Gram $A K A^\top + R$), which reproduces the hand-written conditioning **digit for digit**. (The Julia edition runs this on RxInfer.)
    """
    )
    return


@app.cell
def _(np):
    def gaussian_bp(m0, S0, M, R, y):
        """Sum-product update for a linear-Gaussian tree, moment (gain) form — numerically stable
        for an ill-conditioned prior S0: invert only the observation-space Gram, never S0 itself."""
        G = S0 @ M.T @ np.linalg.inv(M @ S0 @ M.T + R)
        m = m0 + G @ (y - M @ m0)
        S = S0 - G @ M @ S0
        return m, 0.5 * (S + S.T)
    return (gaussian_bp,)


@app.cell
def _(assemble_pde, gaussian_bp, np):
    def rx_pde_solve(prob, n_nodes, n_colloc, l=0.2, s2=1e-8):
        "Solve the BVP with one message-passing update on the operator model."
        d = assemble_pde(prob, n_nodes, n_colloc, l, s2)
        m, P = gaussian_bp(np.zeros(n_nodes), d["K"], d["A"], d["R"], d["c"])
        return dict(X=d["X"], mean=m, std=np.sqrt(np.maximum(np.diag(P), 0.0)))
    return (rx_pde_solve,)


@app.cell
def _(ell_solve, log_s2, n_colloc, n_nodes, prob, rx_pde_solve):
    rx_sol = rx_pde_solve(prob, n_nodes.value, min(n_colloc.value, n_nodes.value - 2),
                          l=ell_solve.value, s2=10.0**log_s2.value)
    return (rx_sol,)


@app.cell
def _(mo, np, rx_sol, sol):
    mo.md(
        f"""
    | quantity | max abs. difference (closed-form collocation vs message passing) |
    |:---|---:|
    | posterior mean over the grid | {np.max(np.abs(rx_sol['mean'] - sol['mean'])):.3g} |
    | posterior std over the grid | {np.max(np.abs(rx_sol['std'] - sol['std'])):.3g} |
    """
    )
    return


@app.cell
def _(PAL, base_layout, go, hex_rgba, np, rx_sol, sol, utrue):
    _up, _lo = sol["mean"] + 2 * sol["std"], sol["mean"] - 2 * sol["std"]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([sol["X"], sol["X"][::-1]]), y=np.concatenate([_up, _lo[::-1]]),
                              fill="toself", fillcolor=hex_rgba(PAL["blue"], 0.15), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="closed form ± 2σ"))
    _fig.add_trace(go.Scatter(x=sol["X"], y=sol["mean"], mode="lines", line=dict(color=PAL["blue"], width=6),
                              opacity=0.35, name="closed form (collocation)"))
    _fig.add_trace(go.Scatter(x=rx_sol["X"], y=rx_sol["mean"], mode="lines",
                              line=dict(color=PAL["green"], width=2, dash="dash"), name="message passing"))
    _fig.add_trace(go.Scatter(x=np.linspace(0, 1, 300), y=utrue(np.linspace(0, 1, 300)), mode="lines",
                              line=dict(color=PAL["black"], width=1.5, dash="dot"), name="true solution"))
    base_layout(_fig, title="Same posterior, two routes", xlabel="x", ylabel="u(x)", legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Where to go from here

    * **Two and three dimensions.** Everything carries over to $\mathcal{L}u = f$ on a domain $\Omega \subset \mathbb{R}^d$: the grid becomes scattered nodes, $\mathcal{L} = -\nabla^2$ a wider stencil, and the solve is still one Gaussian conditioning. The cost is the $\mathcal{O}(m^3)$ of the collocation solve — met with the usual GP remedies.
    * **The kernel is the method.** Choosing the kernel to be the operator's Green's function makes the posterior mean *interpolate the exact solution*, and Owhadi's *gamblets* turn this into fast multiresolution solvers with multigrid-like complexity (Owhadi, 2017).
    * **Uncertainty that composes.** Because the output is a Gaussian belief over the solution *function*, it feeds directly into Bayesian inverse problems — inferring an unknown coefficient while accounting for the solver's own discretization error (Cockayne et al., 2017; 2019).
    * **The whole series, in one graph.** Linear solves (1), quadrature (2), optimization (3), ODE filters (4) and now PDEs are all conjugate Gaussian inference on a factor graph, differing only in which linear operator supplies the observations. The error bars chain.

    ### Exercises

    1. **Calibration of the band.** For $\ell = 0.2$ and a meshless run ($n = 61$ nodes, $m$ collocation points), tabulate the coverage fraction as $m$ grows. For which $m$ does the $2\sigma$ band first contain the true solution everywhere?
    2. **Neumann boundary conditions.** Replace $u(1) = b$ with a flux condition $u'(1) = c$. Add the corresponding row to $A$ and verify the solver still recovers the manufactured solution.
    3. **A variable coefficient.** Solve $-(\kappa(x)\,u')' = f$ for a smooth conductivity $\kappa(x) > 0$. Does the $\mathcal{O}(h^2)$ rate survive?
    4. **The kernel really does wash out.** For full collocation, run the solver with three very different lengthscales and confirm the posterior *mean* is identical while the *covariance* is not. Then reduce the collocation count and watch the mean start to depend on $\ell$.
    5. **Automating the operator.** Assemble the boundary and Laplacian rows from a symbolic description of $\mathcal{L}$ and compare a 2-D Poisson problem against a classical five-point-stencil solve.

    ### References

    * Fasshauer, G. E. (1999). *Solving differential equations with radial basis functions*. Advances in Computational Mathematics, 11, 139–159.
    * Cockayne, J., Oates, C., Sullivan, T., & Girolami, M. (2017). *Probabilistic meshless methods for PDEs and Bayesian inverse problems*. arXiv:1605.07811.
    * Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). *Machine learning of linear differential equations using Gaussian processes*. Journal of Computational Physics, 348, 683–693.
    * Owhadi, H. (2017). *Multigrid with rough coefficients from hierarchical information games*. SIAM Review, 59(1), 99–149.
    * Cockayne, J., Oates, C. J., Sullivan, T. J., & Girolami, M. (2019). *Bayesian probabilistic numerical methods*. SIAM Review, 61(4), 756–789.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press. Free PDF at [probabilistic-numerics.org](https://www.probabilistic-numerics.org).
    """
    )
    return


if __name__ == "__main__":
    app.run()
