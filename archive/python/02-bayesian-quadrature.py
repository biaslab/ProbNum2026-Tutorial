# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""Bayesian Quadrature — Integration as Inference.

ProbNum 2026 tutorial, notebook 2 of 5 (Python / Marimo edition).
"""
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from scipy.special import erf
    return erf, go, mo, np


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
    # Bayesian quadrature: integration as inference

    **ProbNum 2026 tutorial — notebook 2 of 5 (Python / Marimo edition)**

    Probabilistic numerics starts from a simple observation: a numerical algorithm is an *inference engine*. The quantity it computes — a matrix inverse, an integral, a minimizer, an ODE trajectory — is unknown, each evaluation of the problem is *data* about it, and the algorithm's output is really an *estimate*. Once you say it that way, the natural move is to make the estimate a **posterior distribution**: a best guess *plus* a calibrated statement of how wrong it might be, given the finite amount of computation spent.

    In this notebook we work this out for the oldest and cleanest instance: **numerical integration**. Everything stays linear-Gaussian, so every posterior is available in closed form — and, as we show at the end, the same posterior drops out of automatic **message passing on a factor graph** (the Julia edition uses [RxInfer.jl](https://rxinfer.com); here we hand-roll the Gaussian messages in NumPy). That second view is the one that scales to the later notebooks, where the graph becomes a full ODE solver.
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The tutorial series.** This is one of five notebooks, following the five classic problem settings of [probabilistic-numerics.org](https://www.probabilistic-numerics.org):

    1. **Linear algebra** — solving $Ax = b$ as inference over $x$
    2. **Quadrature** — this notebook
    3. **Optimization** — optimization as inference over the minimizer
    4. **Ordinary differential equations** — ODE filters and smoothers
    5. **Partial differential equations** — PDE solving as GP regression on the operator
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

    We want a definite integral of a function that we can only *evaluate*, not integrate symbolically:

    $$
    Z \;=\; \int_a^b f(x)\,\mathrm{d}x .
    $$

    A classical quadrature rule picks nodes $x_1, \dots, x_n$, evaluates $y_i = f(x_i)$, and returns a weighted sum $\hat{Z} = \sum_i w_i\, y_i$ — trapezoid, Simpson, Gauss–Legendre are all of this form, differing only in how the nodes and weights are chosen. They return a **number**. But how wrong is that number? Classical error bounds involve quantities like $\sup_x |f''(x)|$ that we do not know — if we knew $f$ that well, we would not be doing numerical integration.

    **Bayesian quadrature** (O'Hagan, 1991) answers differently: put a prior on the unknown $f$, treat the evaluations $y_i$ as observations, and report the *posterior distribution* of $Z$. The posterior mean turns out to be a weighted sum $\sum_i w_i y_i$ just like a classical rule — but now the weights come from the prior, and the posterior variance is a built-in, model-based error estimate. In fact Diaconis (1988) pointed out that the trapezoidal rule *is* the posterior mean under a particular Gaussian prior: the classical rules were Bayesian all along, they just discarded the uncertainty.

    Here is the integrand we will use throughout — smooth, wiggly, and with no elementary antiderivative:
    """
    )
    return


@app.cell
def _(np):
    def f(x):
        return np.exp(-x**2 / 2) * (1 + np.sin(3 * x))
    a, b = -2.0, 3.0
    return a, b, f


@app.cell
def _(a, b, f, np):
    _xs = np.linspace(a, b, 200_001)
    Z_true = float(np.trapezoid(f(_xs), _xs))   # ground truth by brute-force trapezoid
    return (Z_true,)


@app.cell
def _(PAL, a, b, base_layout, f, go, hex_rgba, np):
    _xs = np.linspace(a - 0.6, b + 0.6, 800)
    _xsab = np.linspace(a, b, 500)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_xsab, y=f(_xsab), fill="tozeroy",
                              fillcolor=hex_rgba(PAL["blue"], 0.15),
                              line=dict(color="rgba(0,0,0,0)"), name="Z (shaded area)"))
    _fig.add_trace(go.Scatter(x=_xs, y=f(_xs), mode="lines", name="f(x)",
                              line=dict(color=PAL["black"], width=2)))
    for _v in (a, b):
        _fig.add_vline(x=_v, line=dict(color=PAL["gray"], dash="dash", width=1))
    base_layout(_fig, title="The integrand and the quantity of interest", xlabel="x", ylabel="f(x)",
                legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. A Gaussian process prior over the integrand

    We model the unknown integrand with a Gaussian process prior, $f \sim \mathcal{GP}(0, k)$, using the squared-exponential kernel

    $$
    k(x, x') \;=\; \sigma_f^2 \exp\!\left( -\frac{(x - x')^2}{2\ell^2} \right),
    $$

    where $\ell$ is the lengthscale (how quickly $f$ wiggles) and $\sigma_f$ its amplitude. This prior says: $f$ is smooth, and values at nearby inputs are strongly correlated.

    Now the key structural fact that makes Bayesian quadrature exact: **integration is a linear functional**, and Gaussian distributions are closed under linear maps. So the function values at the nodes *and the integral itself* are jointly Gaussian under the prior:

    $$
    \begin{bmatrix} f(X) \\ Z \end{bmatrix}
    \sim \mathcal{N}\!\left(
    \mathbf{0},\;
    \begin{bmatrix} K & z \\ z^\top & c \end{bmatrix}
    \right),
    \qquad
    \begin{aligned}
    K_{ij} &= k(x_i, x_j), \\
    z_i &= \textstyle\int_a^b k(x, x_i)\,\mathrm{d}x, \\
    c &= \textstyle\int_a^b\!\!\int_a^b k(x, x')\,\mathrm{d}x\,\mathrm{d}x' .
    \end{aligned}
    $$

    The vector $z$ (the *kernel mean embedding* of the integration measure) and the scalar $c$ (the prior variance of $Z$) are integrals **of the kernel**, not of $f$ — and for the squared-exponential kernel they are available in closed form via the error function:

    $$
    z_i = \sigma_f^2\, \ell \sqrt{\tfrac{\pi}{2}} \left[ \operatorname{erf}\!\left( \tfrac{b - x_i}{\sqrt{2}\,\ell} \right) - \operatorname{erf}\!\left( \tfrac{a - x_i}{\sqrt{2}\,\ell} \right) \right],
    $$

    $$
    c = \sigma_f^2 \left[ 2 L \ell \sqrt{\tfrac{\pi}{2}}\, \operatorname{erf}\!\left( \tfrac{L}{\sqrt{2}\,\ell} \right) + 2 \ell^2 \left( e^{-L^2 / (2\ell^2)} - 1 \right) \right],
    \qquad L = b - a .
    $$

    This is the essential trade in Bayesian quadrature: we exchange one intractable integral (of $f$) for tractable integrals (of $k$).
    """
    )
    return


@app.cell
def _(erf, np):
    def se_kernel(x, xp, l, sf=1.0):
        return sf**2 * np.exp(-(x - xp)**2 / (2 * l**2))

    def kernel_mean(x0, l, sf, a, b):
        "z(x₀) = ∫ₐᵇ k(x, x₀) dx — the kernel mean embedding at node x₀"
        return sf**2 * l * np.sqrt(np.pi / 2) * (
            erf((b - x0) / (np.sqrt(2) * l)) - erf((a - x0) / (np.sqrt(2) * l)))

    def kernel_double_integral(l, sf, a, b):
        "c = ∫ₐᵇ ∫ₐᵇ k(x, x′) dx dx′ — the prior variance of Z"
        L = b - a
        return sf**2 * (2 * L * l * np.sqrt(np.pi / 2) * erf(L / (np.sqrt(2) * l)) +
                        2 * l**2 * (np.exp(-L**2 / (2 * l**2)) - 1))
    return kernel_double_integral, kernel_mean, se_kernel


@app.cell
def _(a, b, kernel_double_integral, kernel_mean, mo, np, se_kernel):
    # Sanity check: closed forms vs brute-force trapezoid integration of the kernel
    _l, _sf, _x0 = 0.7, 1.3, 0.3
    _xs = np.linspace(a, b, 2001)
    _w = (_xs[1] - _xs[0]) * np.concatenate([[0.5], np.ones(len(_xs) - 2), [0.5]])
    _z_num = np.sum(_w * se_kernel(_xs, _x0, _l, _sf))
    _Kmat = se_kernel(_xs[:, None], _xs[None, :], _l, _sf)
    _c_num = _w @ _Kmat @ _w
    _ze = abs(_z_num - kernel_mean(_x0, _l, _sf, a, b))
    _ce = abs(_c_num - kernel_double_integral(_l, _sf, a, b))
    mo.md(f"Closed-form vs numerical kernel integrals — z error: **{_ze:.2e}**, c error: **{_ce:.2e}** (both ≈ 0).")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Conditioning: the Bayesian quadrature posterior

    Given (nearly) noise-free evaluations $y = f(X)$ — we add a tiny $\sigma_n^2$ for numerical stability — conditioning the joint Gaussian above on $y$ gives the posterior over the integral, in closed form:

    $$
    Z \mid y \;\sim\; \mathcal{N}\!\Big(
    \underbrace{z^\top (K + \sigma_n^2 I)^{-1} y}_{\text{posterior mean}},\;
    \underbrace{c - z^\top (K + \sigma_n^2 I)^{-1} z}_{\text{posterior variance}}
    \Big).
    $$

    Two things are worth staring at:

    * The posterior mean is $w^\top y$ with weights $w = (K + \sigma_n^2 I)^{-1} z$. **Bayesian quadrature *is* a quadrature rule** — but the weights are derived from the prior over $f$ rather than from polynomial-exactness conditions.
    * The posterior variance **does not depend on $y$ at all** — only on where we evaluated. This means the uncertainty can be computed (and optimized!) *before* touching the integrand: node placement becomes an experimental-design problem.
    """
    )
    return


@app.cell
def _(kernel_double_integral, kernel_mean, np, se_kernel):
    def bayes_quadrature(X, y, l, sf, a, b, sn2=1e-6):
        K = se_kernel(X[:, None], X[None, :], l, sf) + sn2 * np.eye(len(X))
        z = kernel_mean(X, l, sf, a, b)
        c = kernel_double_integral(l, sf, a, b)
        w = np.linalg.solve(K, z)
        return dict(mu=float(w @ y), sigma=float(np.sqrt(max(c - z @ w, 0.0))),
                    w=w, z=z, c=float(c))
    return (bayes_quadrature,)


@app.cell
def _(np, se_kernel):
    def gp_posterior(xs, X, y, l, sf, sn2=1e-6):
        "GP posterior over f itself, for plotting the belief about the integrand."
        K = se_kernel(X[:, None], X[None, :], l, sf) + sn2 * np.eye(len(X))
        Ks = se_kernel(xs[:, None], X[None, :], l, sf)
        mu = Ks @ np.linalg.solve(K, y)
        v = se_kernel(xs, xs, l, sf) - np.sum(Ks * np.linalg.solve(K, Ks.T).T, axis=1)
        return mu, np.sqrt(np.maximum(v, 0.0))
    return (gp_posterior,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Try it

    The two plots below react to the sliders: the first shows the posterior over the *integrand*, the second the posterior over the *integral*. The amplitude is fixed at $\sigma_f = 1$ throughout.

    Things to try:
    * Increase $n$ and watch both the band around $f$ and the density over $Z$ tighten.
    * Make $\ell$ very small: the prior forgets everything between nodes, and the posterior over $Z$ becomes wide (honest, but slow).
    * Make $\ell$ very large: the posterior becomes narrow — sometimes *too* narrow, excluding the true value. A wrong prior gives a miscalibrated posterior.
    """
    )
    return


@app.cell
def _(mo):
    n_nodes = mo.ui.slider(3, 20, step=1, value=8, label="Number of evaluation nodes n")
    ell = mo.ui.slider(0.1, 2.0, step=0.05, value=0.6, label="Kernel lengthscale ℓ")
    mo.vstack([n_nodes, ell])
    return ell, n_nodes


@app.cell
def _(a, b, bayes_quadrature, ell, f, n_nodes, np):
    X = np.linspace(a, b, n_nodes.value)
    y = f(X)
    bq = bayes_quadrature(X, y, ell.value, 1.0, a, b)
    return X, bq, y


@app.cell
def _(PAL, X, a, b, base_layout, ell, f, go, gp_posterior, hex_rgba, n_nodes, np, y):
    _xs = np.linspace(a, b, 400)
    _mu, _sg = gp_posterior(_xs, X, y, ell.value, 1.0)
    _up, _lo = _mu + 2 * _sg, _mu - 2 * _sg
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([_xs, _xs[::-1]]), y=np.concatenate([_up, _lo[::-1]]),
                              fill="toself", fillcolor=hex_rgba(PAL["blue"], 0.15),
                              line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="± 2σ"))
    _fig.add_trace(go.Scatter(x=_xs, y=_mu, mode="lines", name="GP posterior mean",
                              line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=_xs, y=f(_xs), mode="lines", name="true f (unknown to the method)",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    _fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="evaluations",
                              marker=dict(color=PAL["blue"], size=9, line=dict(color=PAL["white"], width=1.5))))
    base_layout(_fig, title=f"Belief about the integrand (n = {n_nodes.value}, ℓ = {ell.value})",
                xlabel="x", ylabel="f(x)", legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(PAL, Z_true, base_layout, bq, go, hex_rgba, np):
    def _phi(x, mu, sg):
        return np.exp(-((x - mu) / sg)**2 / 2) / (sg * np.sqrt(2 * np.pi))
    _lo = min(bq["mu"] - 4 * bq["sigma"], Z_true) - 0.05
    _hi = max(bq["mu"] + 4 * bq["sigma"], Z_true) + 0.05
    _zs = np.linspace(_lo, _hi, 500)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_zs, y=_phi(_zs, bq["mu"], bq["sigma"]), fill="tozeroy",
                              fillcolor=hex_rgba(PAL["blue"], 0.15),
                              line=dict(color=PAL["blue"], width=2), name="p(Z ∣ evaluations)"))
    _fig.add_vline(x=Z_true, line=dict(color=PAL["black"], dash="dash", width=1.5),
                   annotation_text="true Z")
    base_layout(_fig, title="Belief about the integral", xlabel="Z", ylabel="density",
                legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(Z_true, bq, mo):
    mo.md(
        f"""
    | quantity | value |
    |:---|---:|
    | BQ posterior mean | {bq['mu']:.6f} |
    | BQ posterior std | {bq['sigma']:.3g} |
    | true integral | {Z_true:.6f} |
    | absolute error | {abs(bq['mu'] - Z_true):.3g} |
    | error in posterior std units | {abs(bq['mu'] - Z_true) / bq['sigma']:.3g} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **Calibration.** The last row of the table is the interesting one: a *calibrated* method should typically land within about 2 posterior standard deviations of the truth. If you crank $\ell$ up and see that number explode, the method is confidently wrong — the price of a misspecified prior. Uncertainty quantification is only as good as the model producing it.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. What do we gain? BQ versus Monte Carlo

    Monte Carlo integration converges at rate $\mathcal{O}(n^{-1/2})$ — *regardless* of how smooth the integrand is. That robustness is also a waste: for a smooth $f$, each new evaluation carries information about a whole neighbourhood, and MC ignores it. Bayesian quadrature with a smoothness-encoding kernel exploits it, and on smooth one-dimensional integrands converges dramatically faster.

    The experiment below also plots the **posterior standard deviation** next to the **actual error** — the model's own error estimate tracks the true error, which is the whole point. (This reacts to the $\ell$ slider above.)
    """
    )
    return


@app.cell
def _(Z_true, a, b, bayes_quadrature, ell, f, np):
    def _conv():
        ns = np.arange(2, 25, 2)
        rng = np.random.default_rng(2026)
        bq_err, bq_std, mc_err = [], [], []
        for n in ns:
            Xn = np.linspace(a, b, n)
            r = bayes_quadrature(Xn, f(Xn), ell.value, 1.0, a, b)
            bq_err.append(abs(r["mu"] - Z_true)); bq_std.append(r["sigma"])
            errs = [abs((b - a) * np.mean(f(a + (b - a) * rng.random(n))) - Z_true) for _ in range(1000)]
            mc_err.append(np.mean(errs))
        return ns, np.array(bq_err), np.array(bq_std), np.array(mc_err)
    conv_n, conv_bq_err, conv_bq_std, conv_mc_err = _conv()
    return conv_bq_err, conv_bq_std, conv_mc_err, conv_n


@app.cell
def _(PAL, base_layout, conv_bq_err, conv_bq_std, conv_mc_err, conv_n, go, np):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=conv_n, y=np.maximum(conv_bq_err, 1e-12), mode="lines+markers",
                              line=dict(color=PAL["blue"], width=2),
                              marker=dict(size=7, line=dict(color=PAL["white"], width=1)),
                              name="BQ absolute error"))
    _fig.add_trace(go.Scatter(x=conv_n, y=np.maximum(conv_bq_std, 1e-12), mode="lines",
                              line=dict(color=PAL["blue"], width=2, dash="dash"),
                              name="BQ posterior std (model's own estimate)"))
    _fig.add_trace(go.Scatter(x=conv_n, y=conv_mc_err, mode="lines+markers",
                              line=dict(color=PAL["orange"], width=2),
                              marker=dict(size=7, symbol="diamond", line=dict(color=PAL["white"], width=1)),
                              name="Monte Carlo, mean abs. error (1000 runs)"))
    base_layout(_fig, title="Convergence on a smooth integrand",
                xlabel="number of evaluations n", ylabel="absolute error", legend=dict(x=0.02, y=0.02))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    *The fine print.* The error floor around $10^{-5}$ comes from the jitter $\sigma_n^2 = 10^{-6}$ we add for numerical stability, not from the method. And the comparison flips in high dimension: BQ costs $\mathcal{O}(n^3)$ and inherits the curse of dimensionality through the kernel, while MC's $n^{-1/2}$ is dimension-free. BQ shines when evaluations are *expensive* and the integrand is *smooth and low-dimensional*.

    ## 6. Bayesian quadrature as message passing

    Everything above was bespoke linear algebra. Now the punchline of this tutorial series: the same computation is an instance of **automatic inference on a factor graph**.

    Collect the function values and the integral into one latent vector $g = [f(X);\, Z] \in \mathbb{R}^{n+1}$, with the joint prior we already derived. The evaluations observe the first $n$ components of $g$ through the selection matrix $B = [\,I_n \;\; 0\,]$:

    $$
    g \sim \mathcal{N}(0, K_{\mathrm{aug}}), \qquad
    K_{\mathrm{aug}} = \begin{bmatrix} K & z \\ z^\top & c \end{bmatrix}, \qquad
    y \mid g \sim \mathcal{N}(B g,\ \sigma_n^2 I).
    $$

    As a factor graph:

    ```
        N(0, K_aug) ──▶ (g) ──▶ [ B· ] ──▶ N(·, σₙ²I) ──▶ (y = observed)
                         └─ last component of g is Z ── its posterior marginal is the answer
    ```

    **The integral is literally a node in the graph.** We never wrote the conditioning formulas — sum-product message passing derives them. The graph is a tree and the model is conjugate, so belief propagation is *exact*, and we can check it digit-for-digit against Section 3.
    """
    )
    return


@app.cell
def _(X, bq, ell, n_nodes, np, se_kernel):
    _sn2 = 1e-6
    _K0 = se_kernel(X[:, None], X[None, :], ell.value, 1.0)                 # noise-free kernel matrix
    aug_m0 = np.zeros(n_nodes.value + 1)
    aug_V = np.block([[_K0, bq["z"][:, None]], [bq["z"][None, :], bq["c"]]]) + 1e-10 * np.eye(n_nodes.value + 1)
    aug_B = np.hstack([np.eye(n_nodes.value), np.zeros((n_nodes.value, 1))])
    aug_R = _sn2 * np.eye(n_nodes.value)
    return aug_B, aug_R, aug_V, aug_m0


@app.cell
def _(np):
    def gaussian_bp(m0, S0, M, R, y):
        """Sum-product update for a linear-Gaussian tree, in information (canonical) form:
        the prior factor and the likelihood factor y = M x + N(0, R) meet and multiply."""
        Lam0 = np.linalg.inv(S0)
        eta0 = Lam0 @ m0
        Rinv = np.linalg.inv(R)
        Lam = Lam0 + M.T @ Rinv @ M
        S = np.linalg.inv(Lam)
        return S @ (eta0 + M.T @ Rinv @ y), 0.5 * (S + S.T)
    return (gaussian_bp,)


@app.cell
def _(aug_B, aug_R, aug_V, aug_m0, gaussian_bp, np, y):
    _m, _S = gaussian_bp(aug_m0, aug_V, aug_B, aug_R, y)
    rx_Z_mu = float(_m[-1])                    # last component of g is Z
    rx_Z_sigma = float(np.sqrt(_S[-1, -1]))
    return rx_Z_mu, rx_Z_sigma


@app.cell
def _(bq, mo, rx_Z_mu, rx_Z_sigma):
    mo.md(
        f"""
    | quantity | closed form (Section 3) | message passing |
    |:---|---:|---:|
    | posterior mean of Z | {bq['mu']:.8f} | {rx_Z_mu:.8f} |
    | posterior std of Z | {bq['sigma']:.6g} | {rx_Z_sigma:.6g} |
    """
    )
    return


@app.cell
def _(PAL, Z_true, base_layout, bq, go, np, rx_Z_mu, rx_Z_sigma):
    def _phi(x, mu, sg):
        return np.exp(-((x - mu) / sg)**2 / 2) / (sg * np.sqrt(2 * np.pi))
    _zs = np.linspace(bq["mu"] - 4 * bq["sigma"], bq["mu"] + 4 * bq["sigma"], 400)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_zs, y=_phi(_zs, bq["mu"], bq["sigma"]), mode="lines",
                              line=dict(color=PAL["blue"], width=6), opacity=0.35, name="closed form"))
    _fig.add_trace(go.Scatter(x=_zs, y=_phi(_zs, rx_Z_mu, rx_Z_sigma), mode="lines",
                              line=dict(color=PAL["green"], width=2, dash="dash"), name="message passing"))
    _fig.add_vline(x=Z_true, line=dict(color=PAL["black"], dash="dot", width=1.5), annotation_text="true Z")
    base_layout(_fig, title="Same posterior, two routes", xlabel="Z", ylabel="density",
                legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Where to go from here

    * **Other integration measures.** Replace $\int_a^b f(x)\,dx$ by $\int f(x)\, \mathcal{N}(x \mid \mu, \sigma^2)\, dx$ and the kernel mean $z$ again has a closed form — this is *Bayes–Hermite quadrature* (O'Hagan, 1991), the Bayesian sibling of Gauss–Hermite rules.
    * **Active node selection.** Because the posterior variance is independent of $y$, the next node can be chosen to maximally shrink it — quadrature becomes sequential experimental design.
    * **Better priors.** Warped models such as WSABI encode positivity of the integrand; Matérn kernels encode finite smoothness and change the convergence rate accordingly. The kernel *is* the assumption; choose it like one.
    * **The factor-graph view scales.** The augmented-latent trick — "put the quantity you care about in the graph and let message passing marginalize" — is exactly how notebook 4 turns an ODE initial value problem into a Gaussian state-space model solved by a smoother.

    ### Exercises

    1. **Weights.** Extract `bq["w"]` and compare it to the trapezoidal weights for the same nodes, for small and large $\ell$. Verify numerically that as $\ell \to 0$ the BQ mean tends to (a scaled) Riemann sum.
    2. **Bayes–Hermite.** Derive the kernel mean $z_i$ for a Gaussian integration measure $\mathcal{N}(0, 1)$ with the squared-exponential kernel. Implement it and estimate $\mathbb{E}[f(x)]$ under that measure.
    3. **Active design.** Implement greedy node selection: starting from 2 nodes, repeatedly add the node from a candidate grid that minimizes the posterior variance of $Z$. Compare against equispaced nodes.
    4. **Calibration study.** Fix $n = 8$ and sweep $\ell$ from 0.1 to 2.0. Plot the error-in-std-units against $\ell$. Estimate $\ell$ instead by maximizing the GP log marginal likelihood of $y$ — is the resulting posterior calibrated?
    5. **Unknown noise.** Make the observation-noise variance unknown with an inverse-gamma prior. The model stops being fully conjugate; approximate the posterior over $Z$ and compare.

    ### References

    * O'Hagan, A. (1991). *Bayes–Hermite quadrature*. Journal of Statistical Planning and Inference, 29(3), 245–260.
    * Diaconis, P. (1988). *Bayesian numerical analysis*. In Statistical Decision Theory and Related Topics IV.
    * Rasmussen, C. E., & Ghahramani, Z. (2003). *Bayesian Monte Carlo*. NeurIPS.
    * Briol, F.-X., Oates, C. J., Girolami, M., Osborne, M. A., & Sejdinovic, D. (2019). *Probabilistic integration: a role in statistical computation?* Statistical Science, 34(1), 1–22.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press. Free PDF at [probabilistic-numerics.org](https://www.probabilistic-numerics.org).
    """
    )
    return


if __name__ == "__main__":
    app.run()
