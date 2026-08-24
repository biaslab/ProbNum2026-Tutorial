# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""The Jacobi method.

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
    import plotly.graph_objects as go
    return go, mo, np, sp


@app.cell
def _():
    # Shared palette and small plotting helpers (same as notebook 01).
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
    # The Jacobi method

    **ProbNum 2026 tutorial — companion to notebook 01**

    Notebook 01 spends two paragraphs on the classical solvers and then moves on. This notebook is the long version of one of those paragraphs, for participants who would like the classical method in full before seeing it reappear as message passing.

    The setting is the one from notebook 01: a tiled accelerator, one unknown temperature per tile, and the five-point stencil for $(c - \Delta)u = f$. Tile $i$ dissipates $b_i$, conducts to its four neighbours, and loses heat to the coolant at rate $c$:

    $$
    (4 + c)\,x_i \;-\; \sum_{j \sim i} x_j \;=\; b_i .
    $$

    **The entire Jacobi method is one observation about that equation.** It has one unknown we care about, $x_i$, and four we could pretend to know. So pretend:

    $$
    x_i \;\leftarrow\; \frac{1}{4+c}\Bigl(b_i + \sum_{j\sim i} x_j\Bigr),
    $$

    where the $x_j$ on the right are whatever we currently believe. Every tile does this at once, using its neighbours' *previous* values, and we repeat. That is it. No factorisation, no ordering, nothing global.
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
    ## 1. The splitting

    Write $A = D + R$, with $D$ the diagonal and $R$ everything else. The equation $Ax = b$ says $Dx = b - Rx$, and the Jacobi iteration is what you get by putting the old iterate on the right and the new one on the left:

    $$
    \boxed{\;x^{(t+1)} \;=\; D^{-1}\bigl(b - R\,x^{(t)}\bigr) \;=\; x^{(t)} + D^{-1} r^{(t)},\qquad r^{(t)} = b - Ax^{(t)}.\;}
    $$

    The second form is the useful one: **Jacobi is a correction proportional to the residual**, scaled by the diagonal. Tile $i$ looks at how badly its own equation is violated and moves by that much, divided by its own conductance.

    Two properties, both visible from the formula and both important later:

    * **It is local.** Computing $r_i$ needs $x_j$ for $j \sim i$ and nothing else. On the die, one exchange with the four neighbouring tiles per step.
    * **It is synchronous.** Every tile uses the *previous* round's values. If some tiles updated early, they would be running Gauss–Seidel instead (notebook 01b).
    """
    )
    return


@app.cell
def _(np, sp):
    def jacobi(A, b, iters, omega=1.0, x0=None):
        """Damped Jacobi. omega = 1 is the plain method.

        Returns every iterate, so we can watch the error, not just the last one."""
        A = sp.csr_matrix(A)
        d = A.diagonal()
        x = np.zeros_like(b, dtype=float) if x0 is None else x0.astype(float).copy()
        out = [x.copy()]
        for _ in range(iters):
            x = x + omega * (b - A @ x) / d      # one residual, one divide, no barrier
            out.append(x.copy())
        return out
    return (jacobi,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. When does it converge, and how fast?

    Subtract the fixed point $x_\ast = D^{-1}(b - Rx_\ast)$ from the iteration. The right-hand side $b$ cancels and what is left is a statement about the **error** $e^{(t)} = x^{(t)} - x_\ast$:

    $$
    e^{(t+1)} \;=\; M\,e^{(t)}, \qquad M \;=\; -D^{-1}R \;=\; I - D^{-1}A .
    $$

    So the error is hit by the same matrix every step: $e^{(t)} = M^t e^{(0)}$. This converges from every starting point exactly when the **spectral radius** $\rho(M) = \max_i |\mu_i|$ is less than one, and then the asymptotic rate is $\rho(M)$ per iteration — one digit of accuracy every $-1/\log_{10}\rho$ steps.

    Two sufficient conditions worth remembering, both about $A$ rather than $M$:

    * **Strict diagonal dominance**, $|A_{ii}| > \sum_{j \neq i} |A_{ij}|$, gives $\|M\|_\infty < 1$ directly — every row of $M$ sums to less than one in absolute value. For our stencil, $4 + c > 4$, so *any* screening $c > 0$ is enough.
    * Symmetric positive definite $A$ with $2D - A$ also positive definite.

    For the five-point stencil we can do better than a bound, because we know the eigenvalues exactly. On an $m \times m$ lattice with Dirichlet conditions, $A$ has eigenvalues

    $$
    \lambda_{pq} = c + 4 - 2\cos(p\pi h) - 2\cos(q\pi h), \qquad h = \frac{1}{m+1},\quad p,q = 1,\dots,m,
    $$

    with sine eigenvectors. Since $D = (4+c)I$ is a multiple of the identity, $M = I - A/(4+c)$ has the *same* eigenvectors and eigenvalues

    $$
    \mu_{pq} = \frac{2\cos(p\pi h) + 2\cos(q\pi h)}{4 + c},
    \qquad\text{so}\qquad
    \rho(M) = \frac{4\cos(\pi h)}{4 + c}.
    $$

    As $m$ grows, $\cos(\pi h) \to 1$ and $\rho \to 4/(4+c)$: the rate saturates at a value set by the screening alone. With no screening it goes to $1$ and the method stalls — the same statement as notebook 01's "the round count grows with the diameter when the correlation length is infinite", in classical clothing.
    """
    )
    return


@app.cell
def _(bump_forcing, grid_matrix, np):
    m_grid = 24
    c_grid = 0.4
    A_grid = grid_matrix(m_grid, screening=c_grid)
    b_grid = bump_forcing(m_grid)
    x_star = np.linalg.solve(A_grid.toarray(), b_grid)

    def jacobi_spectrum(m, c):
        "Eigenvalues of the Jacobi iteration matrix, analytically."
        h = 1.0 / (m + 1)
        k = np.arange(1, m + 1)
        cos = 2 * np.cos(k * np.pi * h)
        return (cos[:, None] + cos[None, :]) / (4.0 + c)

    mu_grid = jacobi_spectrum(m_grid, c_grid)
    rho_analytic = np.abs(mu_grid).max()

    _M = np.eye(m_grid ** 2) - np.diag(1.0 / A_grid.diagonal()) @ A_grid.toarray()
    rho_numeric = np.abs(np.linalg.eigvals(_M)).max()
    return (
        A_grid,
        b_grid,
        c_grid,
        jacobi_spectrum,
        m_grid,
        mu_grid,
        rho_analytic,
        rho_numeric,
        x_star,
    )


@app.cell
def _(A_grid, b_grid, jacobi, mo, np, rho_analytic, rho_numeric, x_star):
    _iters = 400
    _xs = jacobi(A_grid, b_grid, _iters)
    _err = np.array([np.linalg.norm(x - x_star) / np.linalg.norm(x_star) for x in _xs])
    _hit = int(np.argmax(_err < 1e-8)) if (_err < 1e-8).any() else None
    _predicted = int(np.ceil(np.log(1e-8) / np.log(rho_analytic)))
    _measured_rate = (_err[-1] / _err[-50]) ** (1 / 50)

    mo.md(
        f"""
    | | value |
    |:--|--:|
    | $\\rho(M)$, analytic $4\\cos(\\pi h)/(4+c)$ | {rho_analytic:.6f} |
    | $\\rho(M)$, numerically from the eigenvalues | {rho_numeric:.6f} |
    | observed contraction over the last 50 steps | {_measured_rate:.6f} |
    | iterations to relative error $10^{{-8}}$, predicted $\\log(10^{{-8}})/\\log\\rho$ | {_predicted} |
    | iterations to relative error $10^{{-8}}$, measured | {_hit} |

    The prediction is asymptotic, so it is a slight over-estimate: the first iterations kill the
    easy error components faster than $\\rho$ suggests. Everything after that is $\\rho$ per step,
    exactly.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. What Jacobi is actually bad at

    The rate $\rho(M)$ is one number, and it hides the interesting structure. Look at the whole spectrum instead. The eigenvalue attached to mode $(p,q)$ is

    $$
    \mu_{pq} = \frac{2\cos(p\pi h) + 2\cos(q\pi h)}{4+c},
    $$

    which is close to $+1$ when both $p,q$ are **small** (smooth error, slowly varying across the die) and close to $-1$ when both are **large** (rough error, alternating tile to tile). Only the middle of the spectrum — error that oscillates in one direction and not the other — is damped quickly.

    That is a problem, because it means plain Jacobi is slow at removing *both* the smoothest and the roughest error. And it is the reason for **damped (weighted) Jacobi**:

    $$
    x^{(t+1)} = x^{(t)} + \omega D^{-1} r^{(t)},
    \qquad
    M_\omega = I - \omega D^{-1}A,
    \qquad
    \mu_{pq}(\omega) = 1 - \omega\bigl(1 - \tfrac{2\cos p\pi h + 2\cos q\pi h}{4+c}\bigr).
    $$

    Taking $\omega < 1$ shifts the whole spectrum away from $-1$. It makes the *overall* rate worse — the smooth end moves closer to $1$ — but it makes the method an excellent **smoother**: after a couple of damped Jacobi sweeps the error is smooth, whatever it started as. A smooth error can be represented on a coarser grid, which is the entire idea of multigrid, and $\omega = 2/3$ is the classical choice because it minimises the worst rough-mode factor.

    Move the slider and watch the two ends of the spectrum trade off.
    """
    )
    return


@app.cell
def _(mo):
    omega_ui = mo.ui.slider(0.1, 1.3, step=0.05, value=1.0,
                            label="damping ω", full_width=True)
    omega_ui
    return (omega_ui,)


@app.cell
def _(PAL, base_layout, c_grid, go, m_grid, mu_grid, np, omega_ui):
    _w = omega_ui.value
    _mu_w = 1 - _w * (1 - mu_grid)
    _k = np.arange(1, m_grid + 1)

    # A one-dimensional slice through the 2-D spectrum: the diagonal p = q.
    _diag = np.diag(_mu_w)
    _rough = np.abs(_mu_w[m_grid // 2:, m_grid // 2:]).max()   # upper half of both directions

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_k, y=_diag, mode="lines+markers", name="μ(p, p)",
                              line=dict(color=PAL["blue"], width=2),
                              marker=dict(size=5)))
    _fig.add_hline(y=0, line=dict(color=PAL["gray"], width=1))
    for _y in (1, -1):
        _fig.add_hline(y=_y, line=dict(color=PAL["black"], width=1, dash="dash"))
    _fig.add_vrect(x0=m_grid // 2, x1=m_grid, fillcolor=PAL["orange"], opacity=0.10,
                   line_width=0, annotation_text="rough half", annotation_position="top left")
    base_layout(_fig, title=(f"Damped-Jacobi eigenvalues along p = q  "
                             f"(ω = {_w:.2f}, c = {c_grid})"),
                xlabel="mode index p", ylabel="μ(ω)", showlegend=False)
    _fig.update_yaxes(range=[-1.15, 1.15])
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(m_grid, mo, mu_grid, np, omega_ui):
    _w = omega_ui.value
    _mu_w = 1 - _w * (1 - mu_grid)
    _rho = np.abs(_mu_w).max()
    _smooth_factor = np.abs(_mu_w[m_grid // 2:, m_grid // 2:]).max()
    mo.md(
        f"""
    | at ω = {_w:.2f} | value |
    |:--|--:|
    | overall rate ρ(M_ω) | {_rho:.4f} |
    | **smoothing factor** — worst rate over the rough half of the spectrum | **{_smooth_factor:.4f}** |

    At ω = 1 the smoothing factor is close to 1 (rough error survives). At ω = 2/3 it is about 1/3:
    three sweeps kill the rough error by an order of magnitude, whatever the grid size. That is
    the property multigrid needs — and it is why notebook 01's §5 mentions GaBP *as a smoother*
    rather than as a standalone solver.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. The distributed reading

    Count what one Jacobi step costs on a machine with one tile per processor:

    | | per iteration |
    |:--|:--|
    | arithmetic | 5 multiply–adds per unknown |
    | communication | one exchange with each neighbour — a *halo exchange* |
    | global operations | **none** |

    Nothing in the update coupled all $n$ unknowns, so nothing had to wait for everything. This is already the structure notebook 01 argues for, and it is why Jacobi survives on hardware where fancier methods struggle.

    There is one honest caveat, and it is the same one every distributed solver has. The *stopping test* $\|b - Ax\| < \varepsilon\|b\|$ **is** a global reduction. If you check it every iteration you have reintroduced the barrier you avoided. The standard fixes are to check every $k$ iterations, to use a fixed iteration count chosen in advance from $\rho$, or — the probabilistic-numerics answer — to give each node a local quantity it can test on its own.

    ## 5. Where this goes in notebook 01

    Notebook 01 derives Gaussian belief propagation, in which each node sends its neighbours two numbers: a mean *and* a precision. Delete the precision — clamp every message precision $P_{ij} := 0$ — and stop excluding the reverse message, and what remains is

    $$
    \mu_i \;=\; \frac{1}{A_{ii}}\Bigl(b_i - \sum_{k \neq i} A_{ki}\mu_k\Bigr),
    $$

    which is the boxed formula at the top of this notebook. **Jacobi is Gaussian belief propagation with the second moment deleted** (Shental et al. 2008, Prop. 16); notebook 01 checks it to $6.7\times 10^{-16}$.

    Read backwards, that is a statement about what Jacobi is missing rather than about what it is. Every tile here is doing inference — it just declines to say how sure it is, and it double-counts the information it has already sent back to whoever sent it.

    ## Exercises

    1. **The screening sweep.** Plot the measured iteration count to $10^{-8}$ against $c \in [0, 2]$ and compare with $\log(10^{-8})/\log\rho$ for $\rho = 4\cos(\pi h)/(4+c)$. Where does the asymptotic prediction stop being useful, and why?
    2. **Grid refinement.** For $c = 0$, measure the iteration count at $m = 8, 16, 32, 64$. Confirm it grows like $m^2$, and explain that using $\rho = \cos(\pi h) \approx 1 - \tfrac12(\pi h)^2$.
    3. **The smoother.** Start from a random error, apply three damped-Jacobi sweeps with $\omega = 2/3$ and $b = 0$, and plot the error before and after. It should look visibly smoother while barely shrinking.
    4. **Optimal damping.** Find the $\omega$ minimising the smoothing factor numerically and compare with $2/3$. Then find the $\omega$ minimising the overall $\rho$ — a different number, for a different purpose.
    5. **Asynchrony.** Update a random 50% of the tiles each sweep. Does it still converge? How does the iteration count change? (This is the question that separates Jacobi from Gauss–Seidel, and notebook 01's flooding schedule from its serial one.)

    ## References

    * Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. Chapter 4.
    * Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*, 2nd ed. SIAM. Chapter 2 is the smoothing-factor argument in full.
    * Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867.
    """
    )
    return


if __name__ == "__main__":
    app.run()
