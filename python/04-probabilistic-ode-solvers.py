# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
"""Probabilistic ODE Solvers — Simulation as Inference.

ProbNum 2026 tutorial, notebook 4 of 5 (Python / Marimo edition).
"""
import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from math import factorial
    return factorial, go, mo, np


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
    # Probabilistic ODE solvers: simulation as inference

    **ProbNum 2026 tutorial — notebook 4 of 5 (Python / Marimo edition)**

    Simulating a dynamical system is the numerical task with the clearest *sequential* structure. A solver for an initial value problem never sees the trajectory $x(t)$; it can only *evaluate the vector field* $f$ at points of its own choosing — and every classical method, from Euler to Runge–Kutta, is a rule for turning those evaluations into a trajectory estimate. The honest description of what the solver knows in between evaluations is a probability distribution.

    In this notebook we rebuild the ODE solver as **Bayesian filtering**: a Gauss–Markov process prior over the trajectory, each vector-field evaluation an observation of it, and the predict–correct loop of a Kalman filter as the solver iteration. This is the *ODE filter* of modern probabilistic numerics (Schober et al., 2019; Tronarp et al., 2019). Along the way, a familiar face: **Heun's classical method drops out as a posterior mean**. And as in every notebook of this series, we finish with a message-passing formulation (the Julia edition's [RxInfer.jl](https://rxinfer.com); here a hand-rolled Gaussian belief propagation) — one solver step is a two-line conjugate model.
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The tutorial series.** One of five notebooks following [probabilistic-numerics.org](https://www.probabilistic-numerics.org): (1) linear algebra, (2) quadrature, (3) optimization, (4) **ODE solvers — this notebook**, (5) PDE solvers.
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

    We want the solution of an initial value problem, $\dot{x}(t) = f(x(t))$, $x(0) = x_0$, $t \in [0, T]$, for a vector field $f$ we can evaluate but not integrate symbolically. A classical single-step solver walks a grid $t_k = kh$, maintaining a point estimate $\hat{x}_k$ and spending one or more evaluations of $f$ per step.

    The probabilistic reading: the trajectory $x(\cdot)$ is a **latent function**, known only through the initial condition and through the fact that it obeys the ODE. Each evaluation of $f$ is a *query* about the slope field. Our running example is the **logistic equation** — nonlinear, and with a closed-form solution so we can grade every posterior against the truth:

    $$
    \dot{x} = r\,x\,(1 - x), \qquad x(0) = x_0 .
    $$
    """
    )
    return


@app.cell
def _(np):
    r_rate, x0, T = 2.5, 0.05, 4.0
    def f(x):
        return r_rate * x * (1 - x)          # the logistic vector field
    def fp(x):
        return r_rate * (1 - 2 * x)          # its derivative, for first-order linearization
    def x_true(t):
        return 1 / (1 + (1 / x0 - 1) * np.exp(-r_rate * t))   # closed-form solution
    return T, f, fp, x0, x_true


@app.cell
def _(PAL, T, base_layout, f, go, np, x0, x_true):
    _fig = go.Figure()
    _gx, _gy = [], []
    _d = 0.11
    for _t in np.linspace(0.05, T - 0.05, 16):
        for _x in np.linspace(-0.05, 1.2, 12):
            _th = np.arctan(f(_x))
            _gx += [_t - _d * np.cos(_th), _t + _d * np.cos(_th), None]
            _gy += [_x - _d * np.sin(_th), _x + _d * np.sin(_th), None]
    _fig.add_trace(go.Scatter(x=_gx, y=_gy, mode="lines", line=dict(color=PAL["gray"], width=1),
                              opacity=0.7, name="slope field", hoverinfo="skip"))
    _ts = np.linspace(0, T, 300)
    _fig.add_trace(go.Scatter(x=_ts, y=x_true(_ts), mode="lines", name="true solution (unknown)",
                              line=dict(color=PAL["black"], width=2.5)))
    _fig.add_trace(go.Scatter(x=[0.0], y=[x0], mode="markers", name="initial value x₀",
                              marker=dict(color=PAL["blue"], size=10, line=dict(color=PAL["white"], width=1.5))))
    base_layout(_fig, title="The vector field is the only thing we can query", xlabel="t", ylabel="x(t)",
                legend=dict(x=0.98, y=0.02, xanchor="right"))
    _fig.update_yaxes(range=[-0.1, 1.25]); _fig.update_xaxes(range=[0, T])
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. The prior: an integrated Wiener process

    For a solver that marches through time we want a GP with **Markov structure**, so inference is a forward recursion with constant cost per step. The standard choice is the $q$-times integrated Wiener process: track the state $z(t) = \big(x(t),\, \dot{x}(t),\, \dots,\, x^{(q)}(t)\big) \in \mathbb{R}^{q+1}$, modelling the $q$-th derivative as a Wiener process. Its transition over a step $h$ is Gaussian **in closed form**: $z(t + h) \mid z(t) \sim \mathcal{N}\!\big(A(h)\, z(t),\, Q(h)\big)$ with

    $$
    A(h)_{ij} = \frac{h^{j-i}}{(j-i)!}, \qquad
    Q(h)_{ij} = \sigma^2 \frac{h^{2q+1-i-j}}{(2q+1-i-j)\,(q-i)!\,(q-j)!} .
    $$

    The state carries the derivative $\dot{x}$ explicitly — exactly the quantity the ODE talks about. $A(h)$ acts on the mean as a **Taylor step**; the prior predicts by extrapolating the local Taylor polynomial and admits uncertainty $Q(h)$ about everything the truncation misses.

    Below are samples from this prior conditioned only on the initial value $x(0) = x_0$ and the initial slope $\dot{x}(0) = f(x_0)$. Higher $q$ gives smoother samples.
    """
    )
    return


@app.cell
def _(factorial, np):
    def iwp_transition(q, h, s2=1.0):
        "Exact discrete-time transition (A, Q) of the q-times integrated Wiener process over a step h."
        d = q + 1
        A = np.array([[h**(j - i) / factorial(j - i) if j >= i else 0.0 for j in range(d)] for i in range(d)])
        Q = s2 * np.array([[h**(2 * q + 1 - i - j) / ((2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j))
                            for j in range(d)] for i in range(d)])
        return A, Q
    return (iwp_transition,)


@app.cell
def _(mo):
    q_prior = mo.ui.slider(1, 3, step=1, value=2, label="Number of derivatives tracked q")
    q_prior
    return (q_prior,)


@app.cell
def _(PAL, T, base_layout, f, go, hex_rgba, iwp_transition, np, q_prior, x0, x_true):
    _rng = np.random.default_rng(2026)
    _Ng = 120; _d = q_prior.value + 1; _h = T / _Ng
    _A, _Q = iwp_transition(q_prior.value, _h)
    _L = np.linalg.cholesky(_Q + 1e-14 * np.eye(_d))
    _tg = np.arange(_Ng + 1) * _h
    _z0 = np.zeros(_d); _z0[0] = x0; _z0[1] = f(x0)
    _P0 = np.zeros((_d, _d))
    for _i in range(2, _d):
        _P0[_i, _i] = 1.0
    _mu = _z0.copy(); _Pk = _P0.copy(); _center = [_mu[0]]; _band = [2 * np.sqrt(_Pk[0, 0])]
    for _ in range(_Ng):
        _mu = _A @ _mu; _Pk = _A @ _Pk @ _A.T + _Q
        _center.append(_mu[0]); _band.append(2 * np.sqrt(max(_Pk[0, 0], 0.0)))
    _center = np.array(_center); _band = np.array(_band)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([_tg, _tg[::-1]]),
                              y=np.concatenate([_center + _band, (_center - _band)[::-1]]), fill="toself",
                              fillcolor=hex_rgba(PAL["blue"], 0.12), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="prior mean ± 2σ (the tangent line)"))
    for _s in range(7):
        _z = _z0 + np.sqrt(np.diag(_P0)) * _rng.standard_normal(_d)
        _path = [_z[0]]
        for _ in range(_Ng):
            _z = _A @ _z + _L @ _rng.standard_normal(_d); _path.append(_z[0])
        _fig.add_trace(go.Scatter(x=_tg, y=_path, mode="lines", line=dict(color=PAL["blue"], width=1.2),
                                  opacity=0.6, showlegend=_s == 0, name="prior samples", hoverinfo="skip"))
    _ts = np.linspace(0, T, 300)
    _fig.add_trace(go.Scatter(x=_ts, y=x_true(_ts), mode="lines", name="true solution",
                              line=dict(color=PAL["black"], width=2, dash="dash")))
    base_layout(_fig, title=f"The IWP({q_prior.value}) prior over the trajectory", xlabel="t", ylabel="x(t)",
                legend=dict(x=0.02, y=0.98))
    _fig.update_yaxes(range=[-5, 5])
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Evaluations as observations

    The solver manufactures its own observations. At every grid point the true solution satisfies the ODE: $0 = \dot{x}(t_k) - f(x(t_k)) = e_1^\top z - f(e_0^\top z)$, where $e_0, e_1$ select the value and derivative. So we *observe the number zero* through a measurement functional of the state (with tiny slack variance $R$). If $f$ is linear this is exact conjugate inference; otherwise we linearize at the predicted mean $\xi = e_0^\top A(h)\, m_{k-1}$:

    $$
    e_1^\top z - f(e_0^\top z) \;\approx\; \underbrace{\big(e_1 - f'(\xi)\, e_0\big)^\top}_{H_k} z \;-\; \underbrace{\big(f(\xi) - f'(\xi)\,\xi\big)}_{b_k},
    $$

    the extended-Kalman-filter move (with the Jacobian, **EKF1**; without it, $H_k = e_1^\top$, **EKF0**). Each solver step is one Kalman predict–correct cycle, and a **Rauch–Tung–Striebel backward pass** afterwards lets every grid point profit from the whole trajectory. The prior scale $\sigma$ is fixed by a **quasi-maximum-likelihood** estimate $\hat{\sigma}^2 = \frac{1}{N}\sum_k r_k^2 / S_k$ accumulated during the forward pass.
    """
    )
    return


@app.cell
def _(iwp_transition, np):
    def ode_filter_smoother(f, fp, x0, T, N, q=2, ekf1=True, R=1e-6, eps0=1e-12):
        "Gaussian ODE filter + RTS smoother with an IWP(q) prior; returns filtered/smoothed beliefs and σ̂²."
        h = T / N; d = q + 1
        A, Q = iwp_transition(q, h)
        e0 = np.eye(d)[0]; e1 = np.eye(d)[1]
        m = np.zeros(d); m[0] = x0; m[1] = f(x0)
        P = eps0 * np.eye(d)
        for i in range(2, d):
            P[i, i] = 1.0
        ms = [m.copy()]; Ps = [P.copy()]; mpred = []; Ppred = []; sse = 0.0
        for _ in range(N):
            mminus = A @ m                                  # predict: a Taylor step, plus uncertainty
            Pminus = A @ P @ A.T + Q
            mpred.append(mminus); Ppred.append(Pminus)
            xi = mminus[0]                                  # linearization point: the predicted value
            H = (e1 - fp(xi) * e0) if ekf1 else e1
            res = f(xi) - mminus[1]                         # innovation: 0 − (ẋ − f(x)) at m⁻
            S = H @ Pminus @ H + R
            K = (Pminus @ H) / S
            m = mminus + K * res                            # correct on the ODE residual
            P = Pminus - np.outer(K, H @ Pminus)
            sse += res**2 / S
            ms.append(m.copy()); Ps.append(P.copy())
        s2 = sse / N                                        # quasi-MLE calibration of the prior scale
        msm = [mk.copy() for mk in ms]; Psm = [Pk.copy() for Pk in Ps]
        for k in range(N - 1, -1, -1):                      # Rauch–Tung–Striebel backward pass
            G = Ps[k] @ A.T @ np.linalg.inv(Ppred[k])
            msm[k] = ms[k] + G @ (msm[k + 1] - mpred[k])
            Psm[k] = Ps[k] + G @ (Psm[k + 1] - Ppred[k]) @ G.T
        return dict(ts=np.arange(N + 1) * h, m=msm, P=Psm, m_filt=ms, P_filt=Ps, s2=s2)
    return (ode_filter_smoother,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Try it

    The plot shows the smoothed posterior over the solution *and* over its derivative — the state tracks both. Bands are $\pm 2$ calibrated standard deviations.

    Things to try:
    * Increase $N$ and watch the band collapse onto the truth.
    * Increase $q$: with the same number of $f$-evaluations the error drops by orders of magnitude — smoothness is computational leverage.
    * At $N = 5$, compare EKF1 with EKF0. EKF0's cheaper linearization destabilizes at coarse steps — but watch the *calibrated band*: the solver knows it is failing.
    """
    )
    return


@app.cell
def _(mo):
    N_steps = mo.ui.slider(5, 80, step=5, value=10, label="Number of solver steps N")
    q_filter = mo.ui.slider(1, 3, step=1, value=1, label="Prior order q")
    mode = mo.ui.dropdown(options={"EKF1 (first-order, uses f′)": "EKF1", "EKF0 (zeroth-order)": "EKF0"},
                          value="EKF1 (first-order, uses f′)", label="Linearization")
    mo.vstack([N_steps, q_filter, mode])
    return N_steps, mode, q_filter


@app.cell
def _(N_steps, T, f, fp, mode, ode_filter_smoother, q_filter, x0):
    sol = ode_filter_smoother(f, fp, x0, T, N_steps.value, q=q_filter.value, ekf1=(mode.value == "EKF1"))
    return (sol,)


@app.cell
def _(PAL, T, base_layout, f, go, hex_rgba, mode, np, q_filter, sol, x_true):
    _sh = np.sqrt(sol["s2"]); _tf = np.linspace(0, T, 300)
    _sx = _sh * np.sqrt(np.maximum([P[0, 0] for P in sol["P"]], 0.0))
    _mx = np.array([m[0] for m in sol["m"]])
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([sol["ts"], sol["ts"][::-1]]),
                              y=np.concatenate([_mx + 2 * _sx, (_mx - 2 * _sx)[::-1]]), fill="toself",
                              fillcolor=hex_rgba(PAL["blue"], 0.15), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="± 2σ (calibrated)"))
    _fig.add_trace(go.Scatter(x=sol["ts"], y=_mx, mode="lines+markers", line=dict(color=PAL["blue"], width=2),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1)), name="posterior mean"))
    _fig.add_trace(go.Scatter(x=_tf, y=x_true(_tf), mode="lines", name="true solution",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    base_layout(_fig, title=f"Belief about the solution ({mode.value}, q = {q_filter.value}, N = {sol['ts'].size - 1})",
                xlabel="t", ylabel="x(t)", legend=dict(x=0.98, y=0.02, xanchor="right"))
    _fig
    return


@app.cell
def _(PAL, T, base_layout, f, go, hex_rgba, np, sol, x_true):
    _sh = np.sqrt(sol["s2"]); _tf = np.linspace(0, T, 300)
    _sd = _sh * np.sqrt(np.maximum([P[1, 1] for P in sol["P"]], 0.0))
    _md = np.array([m[1] for m in sol["m"]])
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([sol["ts"], sol["ts"][::-1]]),
                              y=np.concatenate([_md + 2 * _sd, (_md - 2 * _sd)[::-1]]), fill="toself",
                              fillcolor=hex_rgba(PAL["green"], 0.15), line=dict(color="rgba(0,0,0,0)"),
                              hoverinfo="skip", name="± 2σ (calibrated)"))
    _fig.add_trace(go.Scatter(x=sol["ts"], y=_md, mode="lines+markers", line=dict(color=PAL["green"], width=2),
                              marker=dict(size=6, line=dict(color=PAL["white"], width=1)), name="posterior mean"))
    _fig.add_trace(go.Scatter(x=_tf, y=f(x_true(_tf)), mode="lines", name="true derivative",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    base_layout(_fig, title="Belief about the derivative ẋ(t)", xlabel="t", ylabel="ẋ(t)",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig
    return


@app.cell
def _(mo, np, sol, x_true):
    _mx = np.array([m[0] for m in sol["m"]])
    _maxerr = np.max(np.abs(_mx - x_true(sol["ts"])))
    _sig = np.sqrt(sol["s2"])
    _errT = abs(sol["m"][-1][0] - x_true(sol["ts"][-1])) / np.sqrt(sol["s2"] * max(sol["P"][-1][0, 0], 1e-300))
    mo.md(
        f"""
    | quantity | value |
    |:---|---:|
    | max abs. error of the posterior mean | {_maxerr:.3g} |
    | calibrated prior scale σ̂ | {_sig:.3g} |
    | error at t = T, in calibrated std units | {_errT:.3g} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for**

    * The band is widest where the solution moves fastest — through the logistic transition — because that is where a finite Taylor extrapolation is most wrong, and the residuals say so.
    * The last table row is the calibration check: a value $\lesssim 2$ means the truth sits inside the band. (Values $\ll 1$ mean the band is conservative, the failure mode to prefer.)
    * The derivative panel is not a by-product: the solver returns a joint belief over $x$ and $\dot{x}$, so downstream computations can consume derivative uncertainty directly.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Convergence, calibration — and Heun's method as a posterior mean

    How fast does the posterior mean converge as we spend more evaluations, and does the posterior's own uncertainty track the truth? Each step costs exactly *one* evaluation of $f$ regardless of $q$ — the higher-order information comes from the prior. (The plot below always runs EKF1 on the full grid range; it reacts to nothing.)
    """
    )
    return


@app.cell
def _(T, f, fp, np, ode_filter_smoother, x_true, x0):
    def _conv():
        Ns = [10, 20, 40, 80, 160]
        data = []
        for q in (1, 2, 3):
            errs, sT = [], []
            for N in Ns:
                s = ode_filter_smoother(f, fp, x0, T, N, q=q)
                mx = np.array([m[0] for m in s["m"]])
                errs.append(np.max(np.abs(mx - x_true(s["ts"]))))
                sT.append(np.sqrt(s["s2"] * max(s["P"][-1][0, 0], 0.0)))
            data.append(dict(q=q, err=np.array(errs), sT=np.array(sT)))
        return np.array(Ns), data
    conv_Ns, conv_data = _conv()
    return conv_Ns, conv_data


@app.cell
def _(PAL, base_layout, conv_Ns, conv_data, go, np):
    _fig = go.Figure()
    for _dq, _col in zip(conv_data, [PAL["blue"], PAL["green"], PAL["pink"]]):
        _fig.add_trace(go.Scatter(x=conv_Ns, y=np.maximum(_dq["err"], 1e-16), mode="lines+markers",
                                  line=dict(color=_col, width=2), marker=dict(size=7, line=dict(color=PAL["white"], width=1)),
                                  name=f"posterior mean error — q = {_dq['q']}"))
        _fig.add_trace(go.Scatter(x=conv_Ns, y=_dq["err"][1] * (conv_Ns[1] / conv_Ns)**(_dq["q"] + 1),
                                  mode="lines", line=dict(color=_col, width=1, dash="dot"), showlegend=False))
    _fig.add_trace(go.Scatter(x=conv_Ns, y=np.maximum(conv_data[1]["sT"], 1e-16), mode="lines",
                              line=dict(color=PAL["blue"], width=2, dash="dash"),
                              name="calibrated posterior std at t = T — q = 2"))
    base_layout(_fig, title="Convergence of the posterior mean",
                xlabel="number of steps N (= evaluations of f)", ylabel="max abs. error", legend=dict(x=0.02, y=0.02))
    _fig.update_xaxes(type="log"); _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The dotted references are slopes $N^{-(q+1)}$. Theory guarantees global error $\mathcal{O}(h^q)$; with smoothing and an informed initialization one typically observes $h^{q+1}$, as here. The dashed curve — the solver's *own* error estimate — declines in lockstep with the actual error, which is the whole point of carrying a posterior.

    **The classical method inside.** Take the simplest setting: $q = 1$, exact initial knowledge, no slack, one step of EKF0. The prediction $m^- = (x_0 + h f_0,\; f_0)$ has value component the **explicit Euler step**. The gain works out to $K = (h/2,\; 1)$ and the innovation is $f(x_0 + h f_0) - f_0$, so the posterior mean of the value is

    $$
    x_0 + h f_0 + \tfrac{h}{2}\big(f(x_0 + h f_0) - f_0\big) = x_0 + \tfrac{h}{2}\big(f_0 + f(x_0 + h f_0)\big),
    $$

    which is **Heun's method** — a second-order Runge–Kutta step. Numerically:
    """
    )
    return


@app.cell
def _(f, iwp_transition, np, x0):
    def _heun():
        h = 0.5; f0 = f(x0)
        euler = x0 + h * f0
        heun = x0 + h / 2 * (f0 + f(euler))
        A, Q = iwp_transition(1, h)
        mminus = A @ np.array([x0, f0])                 # predict from a point mass at (x₀, f(x₀))
        res = f(mminus[0]) - mminus[1]
        K = Q[:, 1] / Q[1, 1]                            # Kalman gain: P⁻ = Q, H = e₁ᵀ, R = 0
        m = mminus + K * res
        return abs(mminus[0] - euler), abs(m[0] - heun)
    heun_pred_vs_euler, heun_post_vs_heun = _heun()
    return heun_post_vs_heun, heun_pred_vs_euler


@app.cell
def _(heun_post_vs_heun, heun_pred_vs_euler, mo):
    mo.callout(
        mo.md(
            rf"""
    **Punchline.** Predicted mean vs Euler: **{heun_pred_vs_euler:.3g}**; posterior mean vs Heun: **{heun_post_vs_heun:.3g}**. Euler is the prior prediction, and a single Bayesian update on the ODE residual upgrades it to Heun — one order of accuracy, bought by conditioning. With integrated-Wiener priors, classical Runge–Kutta steps arise as posterior means quite generally (Schober et al., 2014). The classical solvers were Bayesian all along; they just discarded the covariance.
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

    Section 3's predict–correct formulas are exactly the sum-product messages in a linear-Gaussian chain. One solver step is this factor graph:

    ```
        N(mₖ₋₁, Vₖ₋₁) ──▶ (zₖ₋₁) ──▶ [ A(h)· ] ── N(·, Q(h)) ──▶ (zₖ) ──▶ [ Hₖ· ] ── N(·, R) ──▶ (yₖ = bₖ)
    ```

    We iterate it: the transition message turns the previous belief into the *predicted* prior $\mathcal{N}(A m_{k-1},\, A V_{k-1} A^\top + Q)$, and one linear-Gaussian update on the residual corrects it. The linearization $(H_k, b_k)$ is computed from the current belief exactly as in Section 3, so each step's graph is a conjugate tree and the message-passing solver must reproduce the hand-written filter **digit for digit**. (The Julia edition runs this on RxInfer; the RTS smoothing is done in code either way.)
    """
    )
    return


@app.cell
def _(np):
    def gaussian_bp(m0, S0, M, R, y):
        """Sum-product update for a linear-Gaussian tree, in information (canonical) form:
        the (predicted) prior factor and the likelihood factor y = M z + N(0, R) multiply at z."""
        Lam0 = np.linalg.inv(S0)
        Rinv = np.linalg.inv(np.atleast_2d(R))
        M = np.atleast_2d(M)
        Lam = Lam0 + M.T @ Rinv @ M
        S = np.linalg.inv(Lam)
        y = np.atleast_1d(y)
        return (S @ (Lam0 @ m0 + M.T @ Rinv @ y)), 0.5 * (S + S.T)
    return (gaussian_bp,)


@app.cell
def _(gaussian_bp, iwp_transition, np):
    def rx_ode_solve(f, fp, x0, T, N, q=2, ekf1=True, R=1e-6, eps0=1e-12):
        "Solve by iterating the one-step model: predict via the transition, correct via gaussian_bp."
        h = T / N; d = q + 1
        A, Q = iwp_transition(q, h)
        m = np.zeros(d); m[0] = x0; m[1] = f(x0)
        P = eps0 * np.eye(d)
        for i in range(2, d):
            P[i, i] = 1.0
        ms = [m.copy()]; Ps = [P.copy()]
        for _ in range(N):
            m_pred = A @ m                                  # transition message: predicted prior
            P_pred = A @ P @ A.T + Q
            xi = m_pred[0]
            H = np.zeros(d); H[1] = 1.0
            if ekf1:
                H[0] = -fp(xi)
            b = (f(xi) - fp(xi) * xi) if ekf1 else f(xi)    # 0 = ẋ − f(x), linearized: H z = b
            m, P = gaussian_bp(m_pred, P_pred, H, R * np.eye(1), np.array([b]))
            ms.append(m.copy()); Ps.append(P.copy())
        return dict(ts=np.arange(N + 1) * h, m=ms, P=Ps)
    return (rx_ode_solve,)


@app.cell
def _(N_steps, T, f, fp, mode, q_filter, rx_ode_solve, x0):
    rx_sol = rx_ode_solve(f, fp, x0, T, N_steps.value, q=q_filter.value, ekf1=(mode.value == "EKF1"))
    return (rx_sol,)


@app.cell
def _(mo, np, rx_sol, sol):
    _dm = max(np.max(np.abs(rx_sol["m"][k] - sol["m_filt"][k])) for k in range(len(rx_sol["m"])))
    _dP = max(np.max(np.abs(rx_sol["P"][k] - sol["P_filt"][k])) for k in range(len(rx_sol["P"])))
    mo.md(
        f"""
    | quantity | max abs. difference (closed-form filter vs message passing) |
    |:---|---:|
    | filtered means m₀, …, m_N | {_dm:.3g} |
    | filtered covariances P₀, …, P_N | {_dP:.3g} |
    """
    )
    return


@app.cell
def _(PAL, T, base_layout, go, np, rx_sol, sol, x_true):
    _tf = np.linspace(0, T, 300)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=sol["ts"], y=[m[0] for m in sol["m_filt"]], mode="lines",
                              line=dict(color=PAL["blue"], width=6), opacity=0.35, name="closed form (Kalman filter)"))
    _fig.add_trace(go.Scatter(x=sol["ts"], y=[m[0] for m in rx_sol["m"]], mode="lines",
                              line=dict(color=PAL["green"], width=2, dash="dash"), name="message passing"))
    _fig.add_trace(go.Scatter(x=_tf, y=x_true(_tf), mode="lines", line=dict(color=PAL["black"], width=1.5, dash="dot"),
                              name="true solution"))
    base_layout(_fig, title="Same posterior, two routes", xlabel="t", ylabel="x(t)",
                legend=dict(x=0.98, y=0.02, xanchor="right"))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Where to go from here

    * **Real implementations.** Everything here extends to systems $x \in \mathbb{R}^d$ (the state stacks one IWP block per component) and to production quality: square-root filtering, Taylor-mode initialization, adaptive steps, stiff variants. In Julia this is [ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl) (Bosch, 2024).
    * **Adaptivity comes free.** The innovation variance $S_k$ is a model-based local error estimate available *before* the step is accepted — the probabilistic version of embedded error estimators.
    * **A zoo from one lens.** Choices of prior (integrated Wiener, Matérn) and of linearization (EKF0, EKF1, unscented) generate whole families of solvers (Tronarp et al., 2019). Priors are to ODE solvers what kernels were to quadrature.
    * **Solvers that compose.** Because the output is a Gaussian belief rather than a point trajectory, the solver chains into larger inference problems — ODE parameter estimation, latent-force models, control under uncertainty — which is precisely the reading that notebook 5 scales up to partial differential equations.

    ### Exercises

    1. **Calibration across linearizations.** For $q = 2$ and $N \in \{5, 10, 20, 40\}$, tabulate the error-in-std-units for EKF0 and EKF1. Why does knowing $f'$ help the *covariance*?
    2. **Better initialization.** For an autonomous scalar ODE, $\ddot{x}(0) = f'(x_0)\,f(x_0)$. Initialize the $q = 2$ filter with it (zero variance) and measure the effect near $t = 0$.
    3. **Adaptive steps for free.** Use the standardized innovation $r_k^2 / S_k$ to drive a crude step controller; compare error per $f$-evaluation against the fixed grid.
    4. **A linear ODE, exactly.** Solve $\dot{x} = -\lambda x$ with EKF1 and verify the filter computes the *exact* Bayesian posterior under the IWP prior. Why does EKF0 remain an approximation even here?
    5. **Streaming smoothing.** Extend the message-passing loop to also pass backward RTS messages, and confirm the smoothed marginals match the closed-form smoother.

    ### References

    * Schober, M., Duvenaud, D., & Hennig, P. (2014). *Probabilistic ODE solvers with Runge–Kutta means*. NeurIPS.
    * Schober, M., Särkkä, S., & Hennig, P. (2019). *A probabilistic model for the numerical solution of initial value problems*. Statistics and Computing, 29, 99–122.
    * Tronarp, F., Kersting, H., Särkkä, S., & Hennig, P. (2019). *Probabilistic solutions to ODEs as nonlinear Bayesian filtering*. Statistics and Computing, 29, 1297–1315.
    * Kersting, H., Sullivan, T. J., & Hennig, P. (2020). *Convergence rates of Gaussian ODE filters*. Statistics and Computing, 30, 1791–1816.
    * Bosch, N. (2024). *ProbNumDiffEq.jl*. Journal of Open Source Software, 9(101), 7048.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press. Free PDF at [probabilistic-numerics.org](https://www.probabilistic-numerics.org).
    """
    )
    return


if __name__ == "__main__":
    app.run()
