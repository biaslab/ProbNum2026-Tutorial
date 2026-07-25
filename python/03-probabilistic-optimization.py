# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
"""Probabilistic Optimization — Descent as Inference.

ProbNum 2026 tutorial, notebook 3 of 5 (Python / Marimo edition).
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
def _(np):
    PAL = dict(blue="#2a78d6", black="#0b0b0b", green="#008300", orange="#eb6834",
               pink="#e87ba4", gray="#898781", violet="#4a3aa7", white="#fcfcfb")

    def base_layout(fig, title="", xlabel="", ylabel="", **kw):
        fig.update_layout(template="plotly_white", title=title,
                          xaxis_title=xlabel, yaxis_title=ylabel,
                          margin=dict(l=60, r=20, t=50, b=50), **kw)
        return fig

    def ellipse_pts(m, P, nsig=2):
        w, V = np.linalg.eigh((P + P.T) / 2)
        t = np.linspace(0, 2 * np.pi, 160)
        pts = np.asarray(m)[:, None] + nsig * V @ (np.sqrt(np.maximum(w, 0.0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
        return pts[0], pts[1]
    return PAL, base_layout, ellipse_pts


@app.cell
def _(mo):
    mo.md(
        r"""
    # Probabilistic optimization: descent as inference

    **ProbNum 2026 tutorial — notebook 3 of 5 (Python / Marimo edition)**

    An optimizer, like every numerical method in this series, works with a budget of *queries*. It never sees the objective $f$; it can only *evaluate the gradient* $\nabla f$ at points of its own choosing, and every classical method — gradient descent, Newton, BFGS — is a rule for turning those gradient evaluations into an estimate of the minimizer $x_\star$. The honest description of where the minimizer is, given finitely many gradients, is a probability distribution.

    In this notebook we rebuild local optimization as **Bayesian inference over the minimizer**: a Gaussian belief over $x_\star$, each gradient evaluation a linear-Gaussian observation of it, and one optimizer step a single conjugate update. Two familiar faces appear. First, the **inverse Hessian turns out to be a posterior covariance**. Second, the punchline this series owes you once per notebook: **gradient descent and Newton's method both drop out as posterior means** — the choice of curvature model *is* the choice of classical optimizer. As always, we finish by handing the construction to a message-passing formulation (the Julia edition's [RxInfer.jl](https://rxinfer.com); here a hand-rolled Gaussian belief propagation).
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The tutorial series.** One of five notebooks following [probabilistic-numerics.org](https://www.probabilistic-numerics.org): (1) linear algebra, (2) quadrature, (3) **optimization — this notebook**, (4) ODE solvers, (5) PDE solvers.
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

    We want a minimizer of a smooth objective, $x_\star = \arg\min_{x \in \mathbb{R}^d} f(x)$, for an $f$ we can *probe* but not minimize in closed form. A classical local solver maintains a point estimate $x_k$ and, at each step, spends one gradient evaluation to move it: gradient descent takes $x_{k+1} = x_k - \alpha\, g(x_k)$, Newton $x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} g(x_k)$, quasi-Newton something in between.

    The probabilistic reading: the minimizer $x_\star$ is a **latent quantity**, known only through the gradients we have evaluated and through $\nabla f(x_\star) = 0$. Our running example is a convex but non-quadratic 2-D objective — an anisotropic quadratic bowl plus an isotropic quartic — with a **known minimizer at the origin**:

    $$
    f(x) \;=\; \tfrac{1}{2}\, x^\top A\, x \;+\; \tfrac{\beta}{4}\,\|x\|^4,
    \qquad A = \begin{pmatrix} 1 & 0 \\ 0 & \kappa \end{pmatrix}, \qquad x_\star = 0 .
    $$

    The condition number $\kappa$ stretches the bowl; the quartic strength $\beta$ bends the level sets so the curvature changes from point to point ($\beta = 0$ recovers an exact quadratic).
    """
    )
    return


@app.cell
def _(np):
    def make_objective(kappa, beta):
        "Convex 2-D test objective: anisotropic quadratic bowl + isotropic quartic, min at 0."
        A = np.array([[1.0, 0.0], [0.0, float(kappa)]])
        def f(x):
            return 0.5 * (x @ A @ x) + (beta / 4) * (x @ x)**2
        def g(x):
            return A @ x + beta * (x @ x) * x                          # gradient ∇f
        def H(x):
            return A + beta * (x @ x) * np.eye(2) + 2 * beta * np.outer(x, x)   # Hessian ∇²f
        return dict(f=f, g=g, H=H, A=A, xstar=np.zeros(2), fstar=0.0)

    def obj_grid(x1, x2, kappa, beta):
        r2 = x1**2 + x2**2
        return 0.5 * (x1**2 + kappa * x2**2) + (beta / 4) * r2**2

    x_init = np.array([-1.7, 1.35])   # common starting point
    kappa0, beta0 = 8.0, 0.3          # defaults for the display plots
    return kappa0, beta0, make_objective, obj_grid, x_init


@app.cell
def _(PAL, base_layout, beta0, go, kappa0, make_objective, np, obj_grid, x_init):
    _prob = make_objective(kappa0, beta0)
    _xs = np.linspace(-2.2, 2.2, 220); _ys = np.linspace(-2.0, 2.0, 220)
    _X1, _X2 = np.meshgrid(_xs, _ys)
    _Z = obj_grid(_X1, _X2, kappa0, beta0)
    _fig = go.Figure()
    _fig.add_trace(go.Contour(x=_xs, y=_ys, z=_Z, contours_coloring="lines", ncontours=18,
                              colorscale="Viridis", showscale=False, line_width=1, hoverinfo="skip"))
    _gx, _gy = [], []
    _d = 0.16
    for _x in np.linspace(-1.9, 1.9, 11):
        for _y in np.linspace(-1.7, 1.7, 11):
            _g = _prob["g"](np.array([_x, _y])); _u = -_g / (np.linalg.norm(_g) + 1e-9)
            _gx += [_x, _x + _d * _u[0], None]; _gy += [_y, _y + _d * _u[1], None]
    _fig.add_trace(go.Scatter(x=_gx, y=_gy, mode="lines", line=dict(color=PAL["gray"], width=1),
                              opacity=0.7, name="−∇f directions", hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=[x_init[0]], y=[x_init[1]], mode="markers", name="start x₀",
                              marker=dict(color=PAL["blue"], size=10, line=dict(color=PAL["white"], width=1.5))))
    _fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="minimizer x⋆ (unknown)",
                              marker=dict(color=PAL["orange"], size=13, symbol="star")))
    base_layout(_fig, title="The gradient is the only thing we can query", xlabel="x₁", ylabel="x₂")
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. The prior: a Gaussian belief over the minimizer

    Before spending a single gradient we already believe *something* about where $x_\star$ lies. Encode it as a Gaussian, $x_\star \sim \mathcal{N}(m_0, P_0)$, and read the covariance $P_0$ geometrically: its ellipse is the region we think the optimum sits in, and — the point the whole notebook turns on — **$P_0^{-1}$ is a curvature**. The prior we place on the *minimizer* is exactly a prior on the *Hessian* we expect to meet. That correspondence, $P_0 = B^{-1}$ for an assumed curvature $B$, lets a single choice of $B$ select gradient descent, Newton, or quasi-Newton from one and the same update.

    Draw the prior covariance from an assumed curvature $B = \tfrac{1}{s}\,I$ and watch its $\pm 2\sigma$ ellipse over the true level sets — the optimization analogue of the lengthscale in notebook 2.
    """
    )
    return


@app.cell
def _(mo):
    prior_scale = mo.ui.slider(0.25, 3.0, step=0.25, value=1.5,
                               label="Prior scale s (larger ⇒ vaguer belief, shallower assumed bowl)")
    prior_scale
    return (prior_scale,)


@app.cell
def _(PAL, base_layout, beta0, ellipse_pts, go, kappa0, np, obj_grid, prior_scale, x_init):
    _rng = np.random.default_rng(2026)
    _P0 = prior_scale.value * np.eye(2)          # P = B⁻¹ with B = (1/s) I
    _xs = np.linspace(-2.4, 2.4, 200); _ys = np.linspace(-2.2, 2.2, 200)
    _X1, _X2 = np.meshgrid(_xs, _ys)
    _Z = obj_grid(_X1, _X2, kappa0, beta0)
    _fig = go.Figure()
    _fig.add_trace(go.Contour(x=_xs, y=_ys, z=_Z, contours_coloring="lines", ncontours=16,
                              colorscale="Viridis", showscale=False, line_width=1, opacity=0.8, hoverinfo="skip"))
    _ex, _ey = ellipse_pts(x_init, _P0)
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", name="prior ± 2σ",
                              line=dict(color=PAL["blue"], width=2.5)))
    _L = np.linalg.cholesky(_P0)
    _samps = x_init[:, None] + _L @ _rng.standard_normal((2, 40))
    _fig.add_trace(go.Scatter(x=_samps[0], y=_samps[1], mode="markers", showlegend=False,
                              marker=dict(color=PAL["blue"], size=5, opacity=0.5), hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=[x_init[0]], y=[x_init[1]], mode="markers", name="prior mean m₀",
                              marker=dict(color=PAL["blue"], size=9, line=dict(color=PAL["white"], width=1.5))))
    _fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="true minimizer",
                              marker=dict(color=PAL["orange"], size=13, symbol="star")))
    base_layout(_fig, title="The prior belief over x⋆ (± 2σ ellipse)", xlabel="x₁", ylabel="x₂")
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Evaluations as observations

    Nothing arrives from outside — the optimizer manufactures its own observations from the optimality condition. At the minimizer the gradient vanishes, and a first-order expansion about the current iterate $x_k$ gives

    $$
    0 \;=\; \nabla f(x_\star) \;\approx\; \underbrace{\nabla f(x_k)}_{g_k} \;+\; \underbrace{\nabla^2 f(x_k)}_{H_k}\,(x_\star - x_k).
    $$

    Rearranged, this is a **linear-Gaussian observation of $x_\star$**: a measurement matrix $H_k$, a data vector $b_k = H_k x_k - g_k$, and a slack covariance $R$. Conditioning a Gaussian prior $x_\star \sim \mathcal{N}(m_{k-1}, P_{k-1})$ gives one **Kalman correction**,

    $$
    S = H_k P_{k-1} H_k^\top + R, \quad K = P_{k-1} H_k^\top S^{-1}, \quad
    m_k = m_{k-1} - K\, g_k, \quad P_k = P_{k-1} - K H_k P_{k-1}.
    $$

    The one modelling choice left is $H_k$ — the curvature we *claim*. Pick it and read off the optimizer:

    * $H_k = \tfrac{1}{\alpha} I$ ⇒ the step $x_k - \alpha g_k$ — **gradient descent**.
    * $H_k = \nabla^2 f(x_k)$ ⇒ the step $x_k - H_k^{-1} g_k$ — **Newton's method**.
    * $H_k = B_k$, a BFGS estimate from successive gradients — **quasi-Newton**.

    And with the curvature-matched slack $R = H_k$ and a broad prior, $P_k \to H_k^{-1}$ — the **inverse Hessian is the posterior covariance over the minimizer**, the Laplace approximation read as belief propagation.
    """
    )
    return


@app.cell
def _(np):
    def optimize_beliefs(prob, x0, N, method="newton", alpha=0.1, tau=1e6, rho=1.0):
        """Optimize by treating each gradient as a linear-Gaussian observation of x⋆.
        `method`: 'newton' (exact Hessian), 'gradient' (I/α ⇒ gradient descent),
        'quasinewton' (BFGS Hessian estimate). Uses the gain-form Kalman correction."""
        d = len(x0)
        x = np.array(x0, float)
        xs = [x.copy()]
        ms, Ps, Hs, fs, gnorms = [], [], [], [], []
        B = np.eye(d) / alpha                        # BFGS Hessian estimate (if used)
        x_prev = g_prev = None
        for _ in range(N):
            g = prob["g"](x)
            fs.append(prob["f"](x)); gnorms.append(np.linalg.norm(g))
            if method == "quasinewton" and x_prev is not None:
                s = x - x_prev; yk = g - g_prev; sy = s @ yk
                if sy > 1e-12:                       # curvature condition (holds when convex)
                    Bs = B @ s
                    B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(yk, yk) / sy   # BFGS on the Hessian
            H = prob["H"](x) if method == "newton" else (np.eye(d) / alpha if method == "gradient" else B.copy())
            R = rho * H                              # curvature-matched slack
            P0 = tau * np.eye(d)                     # broad prior N(xₖ, τI)
            S = H @ P0 @ H.T + R
            K = P0 @ H.T @ np.linalg.inv(S)
            m = x - K @ g                            # bₖ − H·xₖ = −gₖ
            P = P0 - K @ H @ P0
            ms.append(m.copy()); Ps.append(0.5 * (P + P.T)); Hs.append(H.copy())
            x_prev, g_prev = x.copy(), g.copy()
            x = m; xs.append(x.copy())
        return dict(xs=xs, m=ms, P=Ps, H=Hs, f=np.array(fs), gnorm=np.array(gnorms), method=method)
    return (optimize_beliefs,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Try it

    The first panel shows the iterate path over the level sets with the final posterior ellipse; the second is the optimality gap $f(x_k) - f_\star$ against gradient evaluations.

    Things to try:
    * Switch from **gradient descent** to **Newton**: a handful of steps vs dozens, purely from $H_k$.
    * Crank up $\kappa$: gradient descent zig-zags and stalls (rate $\tfrac{\kappa-1}{\kappa+1}$); Newton and quasi-Newton barely notice.
    * Watch the **quasi-Newton** ellipse start round and stretch to the true inverse Hessian — the optimizer *learning* the geometry.
    """
    )
    return


@app.cell
def _(mo):
    N_steps = mo.ui.slider(2, 40, step=1, value=12, label="Number of gradient evaluations N")
    method_ui = mo.ui.dropdown(
        options={"gradient descent (H = I/α)": "gradient", "Newton (H = ∇²f)": "newton",
                 "quasi-Newton (BFGS)": "quasinewton"},
        value="Newton (H = ∇²f)", label="Curvature model")
    kappa_ui = mo.ui.slider(1.0, 25.0, step=1.0, value=8.0, label="Conditioning κ")
    beta_ui = mo.ui.slider(0.0, 1.5, step=0.1, value=0.3, label="Nonlinearity β")
    alpha_ui = mo.ui.slider(0.02, 0.5, step=0.02, value=0.1, label="Gradient-descent step size α")
    mo.vstack([N_steps, method_ui, kappa_ui, beta_ui, alpha_ui])
    return N_steps, alpha_ui, beta_ui, kappa_ui, method_ui


@app.cell
def _(N_steps, alpha_ui, beta_ui, kappa_ui, make_objective, method_ui, optimize_beliefs, x_init):
    prob_ui = make_objective(kappa_ui.value, beta_ui.value)
    sol = optimize_beliefs(prob_ui, x_init, N_steps.value, method=method_ui.value, alpha=alpha_ui.value)
    return prob_ui, sol


@app.cell
def _(PAL, base_layout, beta_ui, ellipse_pts, go, kappa_ui, np, obj_grid, sol):
    _xs = np.linspace(-2.4, 2.4, 200); _ys = np.linspace(-2.2, 2.2, 200)
    _X1, _X2 = np.meshgrid(_xs, _ys)
    _Z = obj_grid(_X1, _X2, kappa_ui.value, beta_ui.value)
    _px = [p[0] for p in sol["xs"]]; _py = [p[1] for p in sol["xs"]]
    _fig = go.Figure()
    _fig.add_trace(go.Contour(x=_xs, y=_ys, z=_Z, contours_coloring="lines", ncontours=16,
                              colorscale="Viridis", showscale=False, line_width=1, opacity=0.8, hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=_px, y=_py, mode="lines+markers", name="iterates xₖ",
                              line=dict(color=PAL["orange"], width=2),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1))))
    _ex, _ey = ellipse_pts(sol["m"][-1], sol["P"][-1] + 1e-14 * np.eye(2))
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", name="final ± 2σ", line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="x⋆",
                              marker=dict(color=PAL["black"], size=12, symbol="star")))
    base_layout(_fig, title=f"Path & final belief ({sol['method']}, κ = {kappa_ui.value})", xlabel="x₁", ylabel="x₂")
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell
def _(PAL, base_layout, go, np, sol):
    _gap = np.maximum(sol["f"], 1e-16)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.arange(len(_gap)), y=_gap, mode="lines+markers", name=sol["method"],
                              line=dict(color=PAL["green"], width=2),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1))))
    _fig.add_trace(go.Scatter(x=np.arange(len(sol["gnorm"])), y=np.maximum(sol["gnorm"], 1e-16),
                              mode="lines", line=dict(color=PAL["gray"], width=1.5, dash="dash"),
                              name="‖∇f(xₖ)‖ (the solver's own signal)"))
    base_layout(_fig, title="Optimality gap", xlabel="gradient evaluations k", ylabel="f(xₖ) − f⋆",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo, np, sol):
    mo.md(
        f"""
    | quantity | value |
    |:---|---:|
    | distance of final iterate to x⋆ | {np.linalg.norm(sol['xs'][-1]):.3g} |
    | final optimality gap f(x_N) − f⋆ | {sol['f'][-1]:.3g} |
    | final gradient norm ‖∇f(x_N)‖ | {sol['gnorm'][-1]:.3g} |
    | posterior std along x₁, x₂ | {np.sqrt(sol['P'][-1][0,0]):.3g}, {np.sqrt(sol['P'][-1][1,1]):.3g} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for**

    * The posterior ellipse is *narrow* in the steep (large-curvature) direction and *wide* in the flat one — the belief about $x_\star$ is precisely as confident as the local curvature, because its covariance **is** the inverse Hessian.
    * Gradient descent's ellipse is round no matter the landscape: it hard-codes $H = \tfrac{1}{\alpha}I$ and never learns the anisotropy. That mismatch is the same thing as its slow, zig-zagging path.
    * Quasi-Newton needs no Hessian — only gradients — yet its ellipse converges to Newton's. It is *inferring* the curvature from the observations.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Convergence, geometry — and the classical methods as posterior means

    How fast does the posterior mean reach the optimum, and how does that depend on the curvature model? Each step costs exactly *one* gradient regardless of method — the leverage comes from the model. (The plot below runs all three methods; it reacts to nothing.)
    """
    )
    return


@app.cell
def _(make_objective, optimize_beliefs, x_init):
    _prob = make_objective(12.0, 0.3)
    conv = {m: optimize_beliefs(_prob, x_init, 30, method=m, alpha=0.08)
            for m in ("gradient", "newton", "quasinewton")}
    return (conv,)


@app.cell
def _(PAL, base_layout, conv, go, np):
    _fig = go.Figure()
    for _m, _col, _lbl in [("gradient", PAL["blue"], "gradient descent (linear)"),
                           ("quasinewton", PAL["pink"], "quasi-Newton (superlinear)"),
                           ("newton", PAL["green"], "Newton (quadratic)")]:
        _gap = np.maximum(conv[_m]["f"], 1e-16)
        _fig.add_trace(go.Scatter(x=np.arange(len(_gap)), y=_gap, mode="lines+markers", name=_lbl,
                                  line=dict(color=_col, width=2),
                                  marker=dict(size=6, line=dict(color=PAL["white"], width=1))))
    base_layout(_fig, title="Convergence at κ = 12 (one gradient per step)",
                xlabel="gradient evaluations k", ylabel="f(xₖ) − f⋆", legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Three curvature models, three convergence regimes — from one update, differing only in $H_k$. Gradient descent decays linearly (rate set by $\kappa$); Newton is quadratic; quasi-Newton learns the Hessian and lands in between, superlinear. On a genuinely quadratic $\beta = 0$ objective, quasi-Newton reproduces the Hessian within $d$ steps and terminates — the optimization echo of "conjugate gradients solves in $n$ steps".

    **The classical methods inside.** Take one step from $x_0$ with a broad prior anchored there ($P_0 = \tau I,\ \tau \to \infty$). The posterior mean is $m = x_0 - K g_0 \to x_0 - H_0^{-1} g_0$. Choose $H_0 = \tfrac{1}{\alpha}I$ and this is a **gradient-descent step**; choose $H_0 = \nabla^2 f(x_0)$ and it is a **Newton step**. Same update, same gradient; the optimizer's identity is entirely in the prior curvature. Numerically:
    """
    )
    return


@app.cell
def _(beta0, kappa0, make_objective, np, optimize_beliefs, x_init):
    _prob = make_objective(kappa0, beta0); _x0 = x_init
    _g0 = _prob["g"](_x0); _H0 = _prob["H"](_x0); _al = 0.1
    _gd_step = _x0 - _al * _g0
    _newton_step = _x0 - np.linalg.solve(_H0, _g0)
    _gd_bayes = optimize_beliefs(_prob, _x0, 1, method="gradient", alpha=_al)["m"][0]
    _newton_bayes = optimize_beliefs(_prob, _x0, 1, method="newton")["m"][0]
    _cov = optimize_beliefs(_prob, _x0, 1, method="newton")["P"][0]
    mvb_gd = float(np.linalg.norm(_gd_bayes - _gd_step))
    mvb_newton = float(np.linalg.norm(_newton_bayes - _newton_step))
    mvb_cov = float(np.linalg.norm(_cov - np.linalg.inv(_H0)))
    return mvb_cov, mvb_gd, mvb_newton


@app.cell
def _(mo, mvb_cov, mvb_gd, mvb_newton):
    mo.callout(
        mo.md(
            rf"""
    **Punchline.** Posterior mean vs gradient-descent step: **{mvb_gd:.3g}**; vs Newton step: **{mvb_newton:.3g}**; posterior covariance vs inverse Hessian: **{mvb_cov:.3g}**. Gradient descent and Newton are the *same* Bayesian update under different assumed curvatures, and the covariance the update hands back is the inverse Hessian the classical methods never bothered to keep. Quasi-Newton updates such as BFGS are *themselves* posterior means of a Gaussian inference on the Hessian (Hennig & Kiefel, 2013). The classical optimizers were Bayesian all along; they just discarded the covariance.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. The same optimizer as message passing

    Section 3's Kalman correction is exactly the sum-product update in a two-node linear-Gaussian model:

    ```
        N(mₖ₋₁, Pₖ₋₁) ──▶ (x⋆) ──▶ [ Hₖ· ] ── N(·, R) ──▶ (yₖ = bₖ, observed)
    ```

    $x_\star$ is a *static* latent, re-observed each step through a freshly linearized optimality condition $(H_k, b_k)$. Each step's graph is a conjugate tree, so belief propagation is exact and must reproduce the hand-written update **digit for digit**. (The Julia edition runs this on RxInfer; here the same two Gaussian messages in NumPy.) It reacts to the controls of Section 4.
    """
    )
    return


@app.cell
def _(np):
    def gaussian_bp(m0, S0, M, R, y):
        """Sum-product update for a linear-Gaussian tree, in information (canonical) form:
        the prior factor N(m0, S0) and the likelihood factor y = M x + N(0, R) multiply at x."""
        Lam0 = np.linalg.inv(S0)
        Rinv = np.linalg.inv(R)
        Lam = Lam0 + M.T @ Rinv @ M
        S = np.linalg.inv(Lam)
        return S @ (Lam0 @ m0 + M.T @ Rinv @ y), 0.5 * (S + S.T)
    return (gaussian_bp,)


@app.cell
def _(gaussian_bp, np):
    def rx_optimize(prob, x0, N, method="newton", alpha=0.1, tau=1e6, rho=1.0):
        "Iterate the one-step model, swapping the closed-form correction for gaussian_bp."
        d = len(x0)
        x = np.array(x0, float)
        xs = [x.copy()]; ms, Ps = [], []
        B = np.eye(d) / alpha
        x_prev = g_prev = None
        for _ in range(N):
            g = prob["g"](x)
            if method == "quasinewton" and x_prev is not None:
                s = x - x_prev; yk = g - g_prev; sy = s @ yk
                if sy > 1e-12:
                    Bs = B @ s
                    B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(yk, yk) / sy
            H = prob["H"](x) if method == "newton" else (np.eye(d) / alpha if method == "gradient" else B.copy())
            b = H @ x - g
            m, P = gaussian_bp(x, tau * np.eye(d), H, rho * H, b)
            ms.append(m.copy()); Ps.append(P)
            x_prev, g_prev = x.copy(), g.copy()
            x = m; xs.append(x.copy())
        return dict(xs=xs, m=ms, P=Ps)
    return (rx_optimize,)


@app.cell
def _(N_steps, alpha_ui, method_ui, prob_ui, rx_optimize, x_init):
    rx_sol = rx_optimize(prob_ui, x_init, N_steps.value, method=method_ui.value, alpha=alpha_ui.value)
    return (rx_sol,)


@app.cell
def _(mo, np, rx_sol, sol):
    _dm = max(np.max(np.abs(rx_sol["m"][k] - sol["m"][k])) for k in range(len(rx_sol["m"])))
    _dP = max(np.max(np.abs(rx_sol["P"][k] - sol["P"][k])) for k in range(len(rx_sol["P"])))
    mo.md(
        f"""
    | quantity | max abs. difference (closed-form update vs message passing) |
    |:---|---:|
    | posterior means m₁, …, m_N | {_dm:.3g} |
    | posterior covariances P₁, …, P_N | {_dP:.3g} |
    """
    )
    return


@app.cell
def _(PAL, base_layout, beta_ui, go, kappa_ui, np, obj_grid, rx_sol, sol):
    _xs = np.linspace(-2.4, 2.4, 200); _ys = np.linspace(-2.2, 2.2, 200)
    _X1, _X2 = np.meshgrid(_xs, _ys)
    _Z = obj_grid(_X1, _X2, kappa_ui.value, beta_ui.value)
    _fig = go.Figure()
    _fig.add_trace(go.Contour(x=_xs, y=_ys, z=_Z, contours_coloring="lines", ncontours=16,
                              colorscale="Viridis", showscale=False, line_width=1, opacity=0.7, hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=[p[0] for p in sol["xs"]], y=[p[1] for p in sol["xs"]], mode="lines",
                              line=dict(color=PAL["blue"], width=6), opacity=0.35, name="closed form (Kalman update)"))
    _fig.add_trace(go.Scatter(x=[p[0] for p in rx_sol["xs"]], y=[p[1] for p in rx_sol["xs"]], mode="lines+markers",
                              line=dict(color=PAL["green"], width=2, dash="dash"),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1)), name="message passing"))
    _fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="x⋆",
                              marker=dict(color=PAL["black"], size=12, symbol="star")))
    base_layout(_fig, title="Same descent, two routes", xlabel="x₁", ylabel="x₂")
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Where to go from here

    * **Bayesian optimization.** When gradients are unavailable and each evaluation of $f$ is expensive, the same instinct — carry a *distribution* — gives Bayesian optimization: a GP surrogate over $f$ (notebook 2's machinery) and an acquisition function that spends each query where expected improvement is largest (Mockus, 1975; Snoek et al., 2012).
    * **Probabilistic line searches.** The 1-D search inside every descent method can itself be run as inference: a GP on the function-and-gradient along the ray replaces the Wolfe conditions, making descent robust to *noisy* gradients (Mahsereci & Hennig, 2015).
    * **Quasi-Newton as inference, in full.** The BFGS and DFP formulas are *derived* as posterior means of a matrix-variate Gaussian inference on the Hessian, with the secant equations as observations (Hennig & Kiefel, 2013).
    * **Second-order stochastic optimization.** The curvature estimate and its uncertainty feed directly into step-size and trust-region control under gradient noise.

    ### Exercises

    1. **Conditioning and the two rates.** For $\beta = 0$, tabulate the number of gradient evaluations gradient descent and Newton need to reach $f(x_k) - f_\star < 10^{-8}$ for $\kappa \in \{2, 10, 50, 200\}$. Why does the posterior *covariance* explain the difference before you look at the path?
    2. **The inverse Hessian is the covariance.** Verify the Newton posterior covariance $P$ equals $\rho\,\nabla^2 f(x_\star)^{-1}$. Shrink $\rho$ and watch the $\pm 2\sigma$ ellipse tighten. What goes wrong as $\rho \to 0$?
    3. **Quasi-Newton learns the geometry.** Log the BFGS estimate $B_k$ on the $\beta = 0$ problem and measure $\|B_k - A\|$ as $k$ grows. Show it reaches zero within $d = 2$ steps.
    4. **A self-calibrating step-size.** Use $\operatorname{tr} P_k$ as a stopping criterion (halt when the belief about $x_\star$ is tight enough) and compare against a fixed gradient-norm threshold.
    5. **Automatic linearization.** Host the nonlinear map $x \mapsto \nabla f(x)$ inside the model and linearize it automatically, comparing against the hand-linearized Newton iteration.

    ### References

    * Hennig, P., & Kiefel, M. (2013). *Quasi-Newton methods: A new direction*. JMLR, 14, 843–865.
    * Mahsereci, M., & Hennig, P. (2015). *Probabilistic line searches for stochastic optimization*. NeurIPS.
    * Mockus, J. (1975). *On Bayesian methods for seeking the extremum*. Optimization Techniques IFIP.
    * Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms*. NeurIPS.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press. Free PDF at [probabilistic-numerics.org](https://www.probabilistic-numerics.org).
    """
    )
    return


if __name__ == "__main__":
    app.run()
