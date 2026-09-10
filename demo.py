# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""Solving systems of equations with distributed probabilistic numerics.

ProbNum 2026 tutorial — Gaussian belief propagation as a probabilistic linear solver.
"""

import marimo

__generated_with = "0.24.0"
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
    # Shared palette and small plotting helpers.
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
    mo.md(r"""
    # **ProbNum 2026 Tutorial**

    | | |
    |:--|:--|
    | **1. Problem specification** | large sparse structured systems |
    | **2. Classical numerical approach** | Jacobi/Gauss–Seidel, conjugate gradients |
    | **3. Probabilistic numerical approach** | $Ax=b$ as the mean of $\mathcal{N}(A^{-1}b,\, A^{-1})$ |
    | **4. Message-passing version** | Gaussian belief propagation, and scaling  |
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **References**
    * Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867. Extended: [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
    * Bickson, D., Tock, Y., Shental, O., & Dolev, D. (2008). *Polynomial linear programming with Gaussian belief propagation*. Allerton, 895–901.
    * Fanaskov, V. (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear equations*. SIAM J. Sci. Comput., 44(2), A77–A102.
    * Weiss, Y., & Freeman, W. T. (2001). *Correctness of belief propagation in Gaussian graphical models of arbitrary topology*. Neural Computation, 13(10), 2173–2200.
    * Malioutov, D. M., Johnson, J. K., & Willsky, A. S. (2006). *Walk-sums and belief propagation in Gaussian graphical models*. JMLR, 7, 2031–2064.
    * Heskes, T. (2002). *Stable fixed points of loopy belief propagation are local minima of the Bethe free energy*. NeurIPS 15. [proceedings](https://papers.nips.cc/paper/2220-stable-fixed-points-of-loopy-belief-propagation-are-local-minima-of-the-bethe-free-energy).
    * Ihler, A. T., Fisher III, J. W., & Willsky, A. S. (2005). *Loopy belief propagation: convergence and effects of message errors*. JMLR, 6, 905–936. [jmlr.org](https://jmlr.org/papers/v6/ihler05a.html).
    * Johnson, J. K., Bickson, D., & Dolev, D. (2009). *Fixing convergence of Gaussian belief propagation*. IEEE ISIT, 1674–1678. [arXiv:0901.4192](https://arxiv.org/abs/0901.4192).
    * Ruozzi, N., & Tatikonda, S. (2013). *Message-passing algorithms for quadratic minimization*. JMLR, 14, 2287–2314. [jmlr.org](https://jmlr.org/papers/v14/ruozzi13a.html).
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press.
    * Cockayne, J., Oates, C. J., Ipsen, I. C. F., & Girolami, M. (2019). *A Bayesian conjugate gradient method*. Bayesian Analysis, 14(3), 937–1012.
    * Ortiz, J., Pupilli, M., Leutenegger, S., & Davison, A. J. (2020). *Bundle adjustment on a graph processor*. CVPR, 2413–2422. [arXiv:2003.03134](https://arxiv.org/abs/2003.03134).
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Problem specification

    A silicon chip has a few thousand compute tiles on one die, each with its own power counters and temperature sensor. A tile that runs hot has to slow down. But so, often, do its neighbours, because heat spreads sideways through the silicon. To decide whose power to throttle, the control loop needs the steady-state temperature of every tile.

    Tile $i$ dissipates power $b_i$; it conducts heat to the four tiles it touches. It loses heat to the coolant at a rate proportional to how far above ambient it sits. In steady state the three terms balance:

    $$
    \underbrace{c\,x_i}_{\text{lost to the coolant}} \;+\; \underbrace{\sum_{j \sim i}\,(x_i - x_j)}_{\text{conducted to neighbours}} \;=\; \underbrace{b_i}_{\text{dissipated on tile } i}.
    $$

    Approximating the neighbours with a Laplacian yields $(c - \Delta)u = f$. Stack the tile temperatures into $x$ and the dissipated powers into $b$ and the steady-state temperatures are

    $$
    A x = b, \qquad A \in \mathbb{R}^{n\times n}\ \ \text{symmetric},\ \text{sparse},\ \text{one row per tile.}
    $$

    * **$A$ is sparse and structured** Row $i$ couples $x_i$ to the tiles it physically touches. The graph of the matrix is the layout of the chip.
    * **Data is distributed.** $b_i$ is a number tile $i$'s own counters measured, and row $i$ of $A$ is a property of tile $i$'s own package. Nothing was ever assembled anywhere. Assembling it means shipping every tile's telemetry to one place — every control period.
    * **Global synchronisation is the bottleneck.** With a few thousand tiles, a barrier costs more than the arithmetic between barriers, and by the time everyone has checked in, the temperature field has moved.

    The parameter $c \ge 0$ is the screening (reaction) term, i.e., the strength of the coupling to the coolant. Its physical meaning is that it sets how far a hotspot is felt. A well-cooled die will have small hotspots, a poorly cooled one will have hotspots that transfer heat far. It has a probabilistic meaning too, which we will come back to. $c$ affects the **correlation length** of the associated Gaussian field, and indicates how far information has to travel.
    """)
    return


@app.cell
def _(np, sp):
    def chain_matrix(n, diag=2.5):
        "Tridiagonal system: the graph of A is a chain, i.e. a tree."
        off = -np.ones(n - 1)
        return sp.diags([off, diag * np.ones(n), off], [-1, 0, 1], format="csr")

    def lattice_matrix(rows, cols, screening=0.0):
        "Five-point stencil for (c − Δ) on a rows×cols tiling, unknowns numbered row by row."
        d = 4.0 + screening
        T = sp.diags([-np.ones(cols - 1), d * np.ones(cols), -np.ones(cols - 1)],
                     [-1, 0, 1], shape=(cols, cols))
        band = sp.diags([-np.ones(rows - 1), -np.ones(rows - 1)], [-1, 1], shape=(rows, rows))
        A = (sp.kron(sp.eye(rows), T) + sp.kron(band, sp.eye(cols))).tocsr()
        A.eliminate_zeros()          # kron stores explicit zeros; nnz is reported to the reader
        return A

    def grid_matrix(m, screening=0.0):
        "Five-point stencil for (c − Δ) on an m×m lattice: the graph is loopy."
        return lattice_matrix(m, m, screening)

    def bump_forcing(m, centers=((0.3, 0.35), (0.7, 0.65)), width=0.12):
        "Two localised sources on the m×m lattice, flattened row-major."
        g = (np.arange(m) + 0.5) / m
        X, Y = np.meshgrid(g, g, indexing="ij")
        f = np.zeros((m, m))
        for k, (cx, cy) in enumerate(centers):
            f += (1.0 if k % 2 == 0 else -0.8) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
        return f.ravel()

    return bump_forcing, chain_matrix, grid_matrix, lattice_matrix


@app.cell
def _(mo):
    mo.md(r"""
    The picture to keep in mind for the rest of the tutorial: **the matrix *is* a graph**. Node $i$ is the unknown $x_i$; there is an edge $\{i,j\}$ whenever $A_{ij} \neq 0$. Everything the solver will do is expressible as nodes talking along those edges.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Pick a tiling of the die and watch what it does to $A$.
    """)
    return


@app.cell
def _(mo):
    rows_ui = mo.ui.slider(1, 12, step=1, value=4, label="tile rows", full_width=True)
    cols_ui = mo.ui.slider(1, 12, step=1, value=6, label="tile columns", full_width=True)
    mo.vstack([rows_ui, cols_ui])
    return cols_ui, rows_ui


@app.cell
def _(PAL, cols_ui, go, hex_rgba, lattice_matrix, np, rows_ui):
    _rows, _cols = rows_ui.value, cols_ui.value
    _n = _rows * _cols
    _A = lattice_matrix(_rows, _cols)

    # 0 = structural zero, 1 = off-diagonal coupling, 2 = diagonal.
    _nz = np.zeros((_n, _n))
    _nz[np.abs(_A.toarray()) > 0] = 1.0
    np.fill_diagonal(_nz, 2.0)

    _white, _off, _diag = PAL["white"], hex_rgba(PAL["blue"], 0.55), PAL["blue"]
    _fig = go.Figure(go.Heatmap(
        z=_nz, zmin=-0.5, zmax=2.5, showscale=False,
        xgap=1 if _n <= 80 else 0, ygap=1 if _n <= 80 else 0,
        colorscale=[[0.0, _white], [1 / 3, _white], [1 / 3, _off],
                    [2 / 3, _off], [2 / 3, _diag], [1.0, _diag]],
        hovertemplate="A[%{y},%{x}]<extra></extra>",
    ))
    _fig.update_layout(
        template="plotly_white", height=440, margin=dict(l=40, r=20, t=60, b=40),
        title=(f"sparsity pattern of A: {_rows}×{_cols} tiling, "
               f"n = {_n} unknowns, {_A.nnz} non-zeros "
               f"({100 * _A.nnz / _n ** 2:.1f}% dense)"),
    )
    _fig.update_xaxes(title="column j", constrain="domain")
    _fig.update_yaxes(title="row i", autorange="reversed",
                      scaleanchor="x", scaleratio=1, constrain="domain")
    _fig
    return


@app.cell
def _(cols_ui, mo, rows_ui):
    _rows, _cols = rows_ui.value, cols_ui.value
    _n = _rows * _cols
    _within = _rows * (_cols - 1)
    _across = (_rows - 1) * _cols
    mo.md(
        rf"""
    Four off-diagonal bands, and each one is a direction on the die:

    | band | neighbour | edges | why it is there |
    |:--|:--|--:|:--|
    | $\pm 1$ | the tile beside it | {_within} | consecutive numbering follows a row of tiles |
    | $\pm {_cols}$ | the tile above/below it | {_across} | one full row of {_cols} tiles separates them in $i$ |

    The $\pm 1$ band is **torn**: entry $A_{{i,i+1}}$ is missing whenever tile $i$ ends a row, because
    the last tile of one row does not touch the first tile of the next. Tearing accounts for
    {_rows - 1} of the {_n - 1} slots on that band.

    If you change the number of **tile columns**, then the outer bands change. Two tiles at
    $A_{{i,i+{_cols}}}$ are physically adjacent, but a solver marching along the vector meets them
    {_cols} steps apart. If you set either slider to 1, the two bands collapse onto each other and the matrix becomes tridiagonal. This corresponds to a chain. 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Classical numerical approaches

    Three standard ways to solve $Ax = b$ without ever forming $A^{-1}$. All of them are iterative, and all of them touch $A$ only through matrix–vector products, so all of them exploit the sparsity we just looked at.

    **Jacobi.** Split $A = D + R$ into its diagonal and the off-diagonal remainder, and turn the fixed-point equation into a fixed-point iteration

    $$
    x^{(t+1)} = D^{-1}\bigl(b - R\,x^{(t)}\bigr),
    $$

    i.e. *each unknown solves its own equation, assuming its neighbours are right*. This is already local and distributed: tile $i$ needs nothing but the current $x_j$ of the tiles it touches. It converges for every $x^{(0)}$ exactly when the spectral radius of $D^{-1}R$ is below 1, and that spectral radius is also the rate — for a lattice it sits just under 1, so the convergence is slow.

    **Gauss–Seidel.** The same update, with each new value used the moment it exists. Split $A = L + U$ into the lower triangle (diagonal included) and the strict upper triangle and iterate $x^{(t+1)} = L^{-1}\bigl(b - U x^{(t)}\bigr)$. Sweeping in place rather than holding the whole previous iterate roughly squares the Jacobi error per sweep, and costs a sweep order: the update is asynchronous, and the iterates depend on how the tiles are numbered.

    **Conjugate gradients.** For symmetric positive definite $A$, solving $Ax = b$ is minimising

    $$
    q(x) = \tfrac12 x^\top A x - b^\top x ,
    $$

    whose gradient is the negative residual $Ax - b$. CG descends along directions that are $A$-conjugate, $p_i^\top A p_j = 0$ for $i \neq j$, so the $A$-norm error is minimised over the whole Krylov space spanned so far. The price is the step size and the conjugacy correction: every iteration needs $r^\top r$ and $p^\top A p$, two inner products over all $n$ entries. Each one is a global barrier — on a distributed machine every processor waits for every other, twice per iteration.
    """)
    return


@app.cell
def _(np, sp, spla):
    def stationary(A, b, iters, kind="jacobi"):
        "Jacobi (kind='jacobi') or Gauss–Seidel (kind='gs') iterates, from x=0."
        A = sp.csr_matrix(A)
        d = A.diagonal()
        x = np.zeros_like(b, dtype=float)
        out = [x.copy()]
        L = sp.tril(A, format="csr")             # lower triangle, diagonal included
        for _ in range(iters):
            if kind == "jacobi":
                x = x + (b - A @ x) / d          # every node solves its own row
            else:
                x = x + spla.spsolve_triangular(L, b - A @ x, lower=True)
            out.append(x.copy())
        return out

    def conjugate_gradients(A, b, iters):
        "Textbook CG from x=0, returning every iterate."
        x = np.zeros_like(b, dtype=float)
        r = b - A @ x
        p = r.copy()
        rr = r @ r
        out = [x.copy()]
        for _ in range(iters):
            Ap = A @ p
            alpha = rr / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = r @ r
            p = r + (rr_new / rr) * p
            rr = rr_new
            out.append(x.copy())
        return out

    return conjugate_gradients, stationary


@app.cell
def _(mo):
    mo.md(r"""
    Consider a $24 \times 24$ tiling of the die ($n = 576$ unknowns), the five-point stencil of §1 with a small screening term $c$, and two localised heat sources. Drag the slider and watch the three iterations fill the die in.
    """)
    return


@app.cell
def _(bump_forcing, grid_matrix, spla):
    m_grid = 24                       # 24 × 24 tiling of the die, n = 576 unknowns
    screen_grid = 0.4
    A_grid = grid_matrix(m_grid, screening=screen_grid)
    b_grid = bump_forcing(m_grid)
    x_grid = spla.spsolve(A_grid.tocsc(), b_grid)     # reference solution, for the error panels
    return A_grid, b_grid, m_grid, x_grid


@app.cell
def _(A_grid, b_grid, conjugate_gradients, np, stationary):
    K_cls = 60
    cls_names = ["Jacobi", "Gauss–Seidel", "conjugate gradients"]
    cls_iterates = {
        "Jacobi": stationary(A_grid, b_grid, K_cls, "jacobi"),
        "Gauss–Seidel": stationary(A_grid, b_grid, K_cls, "gs"),
        "conjugate gradients": conjugate_gradients(A_grid, b_grid, K_cls),
    }
    _bn = np.linalg.norm(b_grid)
    cls_res = {k: [np.linalg.norm(A_grid @ x - b_grid) / _bn for x in xs]
               for k, xs in cls_iterates.items()}
    return K_cls, cls_iterates, cls_names, cls_res


@app.cell
def _(K_cls, mo):
    iter_slider = mo.ui.slider(0, K_cls, step=1, value=5, label="iterations t", full_width=True)
    field_pick = mo.ui.radio(options=["iterate x⁽ᵗ⁾", "error x⁽ᵗ⁾ − x⋆"],
                             value="iterate x⁽ᵗ⁾", label="show", inline=True)
    mo.vstack([iter_slider, field_pick])
    return field_pick, iter_slider


@app.cell
def _(
    cls_iterates,
    cls_names,
    cls_res,
    field_pick,
    go,
    iter_slider,
    m_grid,
    np,
    x_grid,
):
    _t = iter_slider.value
    _show_err = field_pick.value.startswith("error")
    _lim = float(np.abs(x_grid).max())                # one scale for every panel and every t
    _domains = [(0.00, 0.27), (0.33, 0.60), (0.66, 0.93)]   # right edge left free for the colorbar

    _fig = go.Figure()
    for _i, _name in enumerate(cls_names):
        _x = cls_iterates[_name][_t]
        _z = (_x - x_grid if _show_err else _x).reshape(m_grid, m_grid)
        _ax = "" if _i == 0 else str(_i + 1)
        _fig.add_trace(go.Heatmap(
            z=_z, coloraxis="coloraxis", xaxis=f"x{_ax}", yaxis=f"y{_ax}",
            hovertemplate=f"{_name}<br>%{{z:.3f}}<extra></extra>",
        ))
        _fig.add_annotation(
            x=sum(_domains[_i]) / 2, y=1.06, xref="paper", yref="paper",
            showarrow=False, xanchor="center",
            text=f"<b>{_name}</b><br>‖Ax−b‖/‖b‖ = {cls_res[_name][_t]:.2e}",
        )

    _fig.update_layout(
        template="plotly_white", height=340, margin=dict(l=20, r=20, t=80, b=20),
        title=("after t = {} iterations: {}".format(
            _t, "error x⁽ᵗ⁾ − x⋆" if _show_err else "iterate x⁽ᵗ⁾")),
        coloraxis=dict(colorscale="RdBu", cmin=-_lim, cmax=_lim,
                       colorbar=dict(x=1.0, len=0.85, thickness=14)),
        **{f"xaxis{'' if _j == 0 else _j + 1}": dict(domain=_domains[_j], visible=False)
           for _j in range(3)},
        **{f"yaxis{'' if _j == 0 else _j + 1}": dict(
            visible=False, anchor=f"x{'' if _j == 0 else _j + 1}",
            scaleanchor=f"x{'' if _j == 0 else _j + 1}", scaleratio=1, autorange="reversed")
           for _j in range(3)},
    )
    _fig
    return


@app.cell
def _(K_cls, PAL, base_layout, cls_names, cls_res, go, iter_slider, np):
    _colors = [PAL["blue"], PAL["orange"], PAL["green"]]
    _fig = go.Figure()
    for _name, _c in zip(cls_names, _colors):
        _fig.add_trace(go.Scatter(y=np.maximum(cls_res[_name], 1e-16), x=np.arange(K_cls + 1),
                                  mode="lines", name=_name, line=dict(color=_c, width=2)))
    _fig.add_vline(x=iter_slider.value, line_dash="dot", line_color=PAL["gray"])
    base_layout(_fig, title="relative residual per iteration", xlabel="iteration t",
                ylabel="‖Ax⁽ᵗ⁾ − b‖ / ‖b‖", height=320,
                legend=dict(orientation="h", y=-0.25))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(cls_iterates, cls_names, cls_res, iter_slider, mo, np, x_grid):
    _t = iter_slider.value
    _rows = "\n".join(
        "| {} | {:.2e} | {:.2e} | {} |".format(
            _name, cls_res[_name][_t], np.max(np.abs(cls_iterates[_name][_t] - x_grid)), _cost)
        for _name, _cost in zip(cls_names, ["1 matvec, none", "1 sweep, none (but ordered)",
                                            "1 matvec, **two**"])
    )
    mo.md(
        f"""
    | after t = {_t} iterations | rel. residual | max error | per iteration: work, global barriers |
    |:--|--:|--:|:--|
    {_rows}
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for.** At $t = 1$ the iterate is barely more than the two source bumps — Jacobi's is
    exactly $b_i/A_{ii}$ tile by tile, and CG's is $b$ rescaled by one step size. Jacobi then spreads
    that information one tile per iteration — the front you see creeping outward is literally the graph
    distance covered so far. Gauss–Seidel spreads it faster in the sweep direction. Conjugate gradients is not local at all: after a handful of iterations its iterate already has global structure, because each step mixes information across
    the whole die through the two inner products.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Probabilistic numerical approach

    The usual probabilistic reading of a linear solver puts a Gaussian prior on $x$, treats each matrix–vector product as a linear observation $s_i^\top A x_\ast = s_i^\top b$, and conditions. It inherits a dense $n \times n$ covariance and a global policy for choosing $s_i$.

    For symmetric positive definite $A$, define

    $$
    q(x) = \tfrac12 x^\top A x - b^\top x, \qquad
    p(x) \;\propto\; \exp\bigl(-q(x)\bigr) \;=\; \exp\bigl(-\tfrac12 x^\top A x + b^\top x\bigr).
    $$

    Completing the square gives

    $$
    \boxed{\;p(x) \;=\; \mathcal{N}\bigl(x;\ A^{-1}b,\ A^{-1}\bigr).\;}
    $$

    So the solution vector *is* the mean of a Gaussian whose precision matrix is $A$ itself (Shental et al. 2008, Prop. 8). Solving a linear system and computing the marginal means of a Gaussian Markov random field are the same problem:

    | linear algebra | probabilistic inference |
    |:--|:--|
    | matrix $A$ | precision (information) matrix |
    | right-hand side $b$ | natural-parameter mean $A\mu$ |
    | sparsity pattern of $A$ | conditional independence graph |
    | solution $x_\ast = A^{-1}b$ | vector of marginal means $\mu_i$ |
    | diagonal of $A^{-1}$ | marginal variances $\sigma_i^2$ |
    | $1/A_{ii}$ | *conditional* variance of $x_i$ given its neighbours |

    This leads us to two observations

    1. **The uncertainty is free-standing.** We did not choose a prior and we are not modelling rounding error. The Gaussian is a re-encoding of the problem itself, and its marginal variances $(A^{-1})_{ii}$ are the quantity a statistician would want anyway when $A$ is a posterior precision (Gaussian process regression, GMRF models, Kalman smoothing, bundle adjustment).

       Back on the die, that quantity is not a statistical abstraction either: $(A^{-1})_{ij}$ is the temperature rise at tile $i$ per unit of power dissipated at tile $j$ — the thermal impedance of the chip, which is what a thermal engineer would have measured. Its diagonal, the marginal variance, is how hot tile $i$ gets from its own watt *once the rest of the die has been allowed to warm up in response*. The conditional variance $1/A_{ii}$ is the same number computed with every neighbour pinned to ambient: the answer a tile would give if it believed it were the only warm thing on the chip. The gap between those two is real, physical, and — as §4 will show — is precisely what the messages carry.
    2. **Locality is now structural.** $A_{ij} = 0$ means $x_i \perp x_j \mid x_{\text{rest}}$. The graph of the matrix is the conditional independence graph of the belief, so an inference algorithm that only exchanges information along edges is *automatically* a solver that only communicates along the sparsity pattern.

    The last row of the table is the seed of the entire algorithm. A node that knows only its own equation knows the conditional variance $1/A_{ii}$; to upgrade it to the marginal variance $(A^{-1})_{ii}$ it has to hear from the rest of the graph.
    """)
    return


@app.cell
def _(PAL, base_layout, go, np):
    # The Gaussian view in two dimensions: p(x) ∝ exp(−½ xᵀAx + bᵀx) is centred at A⁻¹b.
    _A = np.array([[3.0, 1.4], [1.4, 2.0]])
    _b = np.array([1.0, 2.0])
    _x = np.linalg.solve(_A, _b)
    _g = np.linspace(-1.2, 1.8, 160)
    _X, _Y = np.meshgrid(_g, _g, indexing="ij")
    _Q = 0.5 * (_A[0, 0] * _X ** 2 + 2 * _A[0, 1] * _X * _Y + _A[1, 1] * _Y ** 2) - (_b[0] * _X + _b[1] * _Y)

    _fig = go.Figure()
    _fig.add_trace(go.Contour(x=_g, y=_g, z=np.exp(-(_Q - _Q.min())).T, showscale=False,
                              colorscale="Blues", contours=dict(showlines=False), name="p(x)"))
    _fig.add_trace(go.Scatter(x=[_x[0]], y=[_x[1]], mode="markers+text", text=["  A⁻¹b"],
                              textposition="middle right", textfont=dict(color=PAL["black"]),
                              marker=dict(color=PAL["orange"], size=14, symbol="star",
                                          line=dict(color=PAL["white"], width=1)),
                              name="solution = mean"))
    base_layout(_fig, title="p(x) ∝ exp(−½ xᵀA x + bᵀx) = 𝒩(A⁻¹b, A⁻¹)", xlabel="x₁", ylabel="x₂",
                legend=dict(x=0.02, y=0.98))
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig.update_layout(height=430)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### BayesCG: conditioning on matrix–vector products

    The Gaussian above is the answer, but it can also serve as the **prior**. Take $p_0(x) = \mathcal{N}(0,\, A^{-1})$ — the belief of somebody who knows $A$ but has not yet looked at $b$ — and let the solver observe the right-hand side a few directions at a time. Picking a search direction $s_i$ and computing the matrix–vector product returns $s_i^\top A x_\ast = s_i^\top b$, a noiseless linear observation of $x_\ast$, so after $m$ of them the posterior is Gaussian again,

    $$
    \mu_m = \Sigma_0 A^\top S_m \bigl(S_m^\top A \Sigma_0 A^\top S_m\bigr)^{-1} S_m^\top b,
    \qquad
    \Sigma_m = \Sigma_0 - \Sigma_0 A^\top S_m \bigl(S_m^\top A \Sigma_0 A^\top S_m\bigr)^{-1} S_m^\top A \Sigma_0,
    $$

    with $S_m = [\,s_1 \cdots s_m\,]$. That is **BayesCG** (Cockayne et al. 2019). Taking $\Sigma_0 = A^{-1}$ — the distribution we just derived — collapses it: write the directions $A$-orthonormal, $\hat{s}_i^\top A \hat{s}_j = \delta_{ij}$, and

    $$
    \mu_m = \sum_{i \le m} \hat{s}_i\,(\hat{s}_i^\top b),
    \qquad
    \Sigma_m = A^{-1} - \sum_{i \le m} \hat{s}_i \hat{s}_i^\top .
    $$

    Every direction resolves exactly one dimension of the belief and leaves the other $n - m$ untouched. Take the directions to be the residuals $b - A\mu_{i-1}$ and the posterior mean *is* the conjugate-gradients iterate of §2; take them any other way and the same bookkeeping gives a slower solver.

    A smaller die here — a $10 \times 10$ tiling, $n = 100$ — so that the budget of $n$ directions can actually be exhausted.
    """)
    return


@app.cell
def _(np):
    def bayescg(A, prior_var, b, iters, directions="residual", seed=0):
        """BayesCG with prior 𝒩(0, A⁻¹), directions A-orthonormalised as they arrive.

        directions : 'residual'   — s = b − Aμ, which reproduces conjugate gradients
                     'coordinate' — s = e₁, e₂, … , one tile at a time
                     'random'     — s ~ 𝒩(0, I)
        Returns the posterior mean and marginal variances after 0, 1, … iters directions,
        plus how many directions were actually usable (the Krylov space can run out).
        """
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        mu = np.zeros(n)
        var = prior_var.copy()
        S = []                                            # A-orthonormal directions so far
        mus, vars_, used = [mu.copy()], [var.copy()], [0]
        for k in range(iters):
            if directions == "residual":
                s = b - A @ mu
            elif directions == "coordinate":
                s = np.zeros(n)
                s[k % n] = 1.0
            else:
                s = rng.standard_normal(n)
            for _ in range(2):                            # A-conjugate, with re-orthogonalisation
                for u in S:
                    s = s - (u @ (A @ s)) * u
            q = float(s @ (A @ s))
            if np.isfinite(q) and q > 1e-14:              # otherwise the direction is spent
                s = s / np.sqrt(q)
                S.append(s)
                mu = mu + s * (s @ b)                     # one dimension resolved …
                var = var - s ** 2                        # … and removed from the belief
            mus.append(mu.copy())
            vars_.append(np.maximum(var, 0.0))
            used.append(len(S))
        return dict(mus=mus, vars=vars_, used=used)

    return (bayescg,)


@app.cell
def _(bayescg, bump_forcing, grid_matrix, np, spla):
    m_bcg = 10                                            # 10 × 10 tiling, n = 100 unknowns
    A_bcg = grid_matrix(m_bcg, screening=0.4)
    b_bcg = bump_forcing(m_bcg)
    x_bcg = spla.spsolve(A_bcg.tocsc(), b_bcg)
    n_bcg = m_bcg ** 2
    pv_bcg = np.diag(np.linalg.inv(A_bcg.toarray()))      # prior marginal variances, diag(A⁻¹)
    dir_names = ["residual (= CG)", "coordinate", "random"]
    bcg_runs = {
        "residual (= CG)": bayescg(A_bcg, pv_bcg, b_bcg, n_bcg, "residual"),
        "coordinate": bayescg(A_bcg, pv_bcg, b_bcg, n_bcg, "coordinate"),
        "random": bayescg(A_bcg, pv_bcg, b_bcg, n_bcg, "random"),
    }

    def a_norm(A, v):
        return float(np.sqrt(abs(v @ (A @ v))))

    bcg_err = {k: [a_norm(A_bcg, x_bcg - mu) / a_norm(A_bcg, x_bcg) for mu in r["mus"]]
               for k, r in bcg_runs.items()}
    return (
        A_bcg,
        b_bcg,
        bcg_err,
        bcg_runs,
        dir_names,
        m_bcg,
        n_bcg,
        pv_bcg,
        x_bcg,
    )


@app.cell
def _(dir_names, mo, n_bcg):
    dir_pick = mo.ui.radio(options=dir_names, value=dir_names[0], label="search directions",
                           inline=True)
    m_slider = mo.ui.slider(0, n_bcg, step=1, value=8, label="search directions m", full_width=True)
    mo.vstack([m_slider, dir_pick])
    return dir_pick, m_slider


@app.cell
def _(bcg_runs, dir_pick, go, m_bcg, m_slider, np, pv_bcg, x_bcg):
    _r = bcg_runs[dir_pick.value]
    _m = m_slider.value
    _mu = _r["mus"][_m].reshape(m_bcg, m_bcg)
    _sd = np.sqrt(_r["vars"][_m]).reshape(m_bcg, m_bcg)

    _fig = go.Figure()
    _fig.add_trace(go.Heatmap(z=_mu, colorscale="RdBu", zmid=0, colorbar=dict(x=0.44, len=0.9),
                              zmin=float(x_bcg.min()), zmax=float(x_bcg.max()),
                              hovertemplate="μ = %{z:.3f}<extra></extra>"))
    _fig.add_trace(go.Heatmap(z=_sd, colorscale="Viridis", xaxis="x2", yaxis="y2",
                              colorbar=dict(x=1.0, len=0.9),
                              zmin=0.0, zmax=float(np.sqrt(pv_bcg).max()),
                              hovertemplate="σ = %{z:.3f}<extra></extra>"))
    _fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=40, r=20, t=60, b=40),
        title=f"after m = {_m} directions:  posterior mean μₘ (left), posterior std √diag(Σₘ) (right)",
        xaxis=dict(domain=[0.0, 0.42], visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, autorange="reversed"),
        xaxis2=dict(domain=[0.56, 0.98], visible=False),
        yaxis2=dict(visible=False, anchor="x2", scaleanchor="x2", scaleratio=1, autorange="reversed"),
    )
    _fig
    return


@app.cell
def _(
    PAL,
    base_layout,
    bcg_err,
    bcg_runs,
    dir_pick,
    go,
    m_slider,
    n_bcg,
    np,
    pv_bcg,
):
    _r = bcg_runs[dir_pick.value]
    _sd = np.array([np.sqrt(v).mean() for v in _r["vars"]]) / np.sqrt(pv_bcg).mean()
    _m = np.arange(n_bcg + 1)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_m, y=np.maximum(bcg_err[dir_pick.value], 1e-16), mode="lines",
                              name="relative A-norm error", line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=_m, y=_sd, mode="lines", yaxis="y2",
                              name="mean posterior std (÷ prior)",
                              line=dict(color=PAL["orange"], width=2)))
    _fig.add_vline(x=m_slider.value, line_dash="dot", line_color=PAL["gray"])
    base_layout(_fig, title="what the mean knows vs. what the belief admits",
                xlabel="search directions m", ylabel="relative A-norm error", height=330,
                legend=dict(orientation="h", y=-0.28),
                yaxis2=dict(title="posterior std ÷ prior std", overlaying="y", side="right",
                            range=[0, 1.05], showgrid=False))
    _fig.update_yaxes(type="log")
    _fig
    return


@app.cell
def _(
    A_bcg,
    b_bcg,
    bcg_err,
    bcg_runs,
    conjugate_gradients,
    dir_pick,
    m_slider,
    mo,
    np,
    pv_bcg,
    x_bcg,
):
    _r = bcg_runs[dir_pick.value]
    _m = m_slider.value
    _mu, _var = _r["mus"][_m], _r["vars"][_m]
    _sd = np.sqrt(_var)
    _extra = ""
    if dir_pick.value.startswith("residual"):
        _K = min(_r["used"][-1], 20)
        _cg = conjugate_gradients(A_bcg, b_bcg, _K)
        _gap = max(np.max(np.abs(_r["mus"][_i] - _cg[_i])) for _i in range(_K + 1))
        _extra = (f"\n| max difference between $\\mu_m$ and the CG iterate, first {_K} steps "
                  f"| **{_gap:.1e}** |")
    mo.md(
        f"""
    | after m = {_m} directions | value |
    |:--|--:|
    | directions actually usable | {_r['used'][_m]} of {_m if _m else 0} |
    | relative error in the $A$-norm | {bcg_err[dir_pick.value][_m]:.2e} |
    | max error in the mean | {np.max(np.abs(_mu - x_bcg)):.2e} |
    | mean posterior std, as a fraction of the prior | {_sd.mean() / np.sqrt(pv_bcg).mean():.3f} |
    | tiles whose truth lies within ±2σ of μₘ | {int(np.sum(np.abs(_mu - x_bcg) <= 2 * _sd))} of {len(x_bcg)} |{_extra}
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The mean finishes long before the belief does.** With residual directions the posterior mean
    reproduces conjugate gradients to the last bit — the known BayesCG–CG correspondence — and it is
    accurate to $10^{-7}$ in $A$-norm after about 20 directions. The posterior standard deviation at that
    point has barely moved: 100 dimensions, 20 of them resolved, so roughly 80 % of the prior spread is
    still there. Keep dragging and CG runs out of Krylov space entirely (the "directions actually usable"
    row stops climbing) while the belief still reports the variance of every direction it never looked in.
    The credible intervals are not wrong — every tile stays inside $\pm 2\sigma$ — they are wildly
    conservative, and calibrating them is an active line of work.

    Switch to coordinate or random directions and the opposite happens: the variance falls at the same
    steady rate as before, all the way to zero at $m = n$, but the mean now needs nearly all $n$
    directions to get there. The uncertainty is tracking *how much of the space has been probed*, not
    *how wrong the estimate is*. Section 4.5 meets the mirror image of this — a solver that is
    over-confident rather than over-cautious.

    Note also what the prior cost: $\Sigma_0 = A^{-1}$ and every $\Sigma_m$ is a dense $n \times n$
    object, and choosing $s_i$ is a global decision. That is what §4 gets rid of.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The message-passing version

    ### 4.1 The factor graph

    Write the density as a product of one factor per node and one per edge:

    $$
    p(x) \;\propto\; \prod_{i} \phi_i(x_i) \prod_{\{i,j\}} \psi_{ij}(x_i,x_j),
    \qquad
    \phi_i(x_i) = \exp\!\bigl(b_i x_i - \tfrac12 A_{ii} x_i^2\bigr),
    \qquad
    \psi_{ij}(x_i,x_j) = \exp(-x_i A_{ij} x_j).
    $$

    The self-factor $\phi_i$ is row $i$'s own equation — it is $\mathcal{N}(x_i;\, b_i/A_{ii},\, 1/A_{ii})$, exactly the "solve my equation ignoring the coupling" belief that Jacobi starts from. The edge factor $\psi_{ij}$ is the coupling. **Every quantity in the factor graph is an entry of $A$ or $b$ that node $i$ already owns.**

    ### 4.2 The messages

    Sum-product on this graph: the message from $i$ to $j$ is

    $$
    m_{i\to j}(x_j) \;\propto\; \int \psi_{ij}(x_i,x_j)\, \phi_i(x_i) \!\!\prod_{k \in N(i)\setminus j}\!\! m_{k\to i}(x_i)\, \mathrm{d}x_i .
    $$

    Products of Gaussians are Gaussian and Gaussian integrals are Gaussian, so each message is carried by **two scalars**: a precision $P_{ij}$ and a mean $\mu_{ij}$. Writing $P_{i\setminus j}$ for the precision node $i$ has accumulated *excluding* what $j$ told it,

    $$
    P_{i\setminus j} = A_{ii} + \!\!\sum_{k\in N(i)\setminus j}\!\! P_{ki},
    \qquad
    \mu_{i\setminus j} = \frac{1}{P_{i\setminus j}}\Bigl(b_i + \!\!\sum_{k\in N(i)\setminus j}\!\! P_{ki}\mu_{ki}\Bigr),
    $$

    the outgoing message is

    $$
    \boxed{\;P_{ij} = -\frac{A_{ij}^2}{P_{i\setminus j}}, \qquad
    \mu_{ij} = \frac{P_{i\setminus j}\,\mu_{i\setminus j}}{A_{ij}}\;}
    $$

    and the belief at node $i$, using *all* incoming messages, is

    $$
    P_i = A_{ii} + \sum_{k\in N(i)} P_{ki},
    \qquad
    \mu_i = \frac{1}{P_i}\Bigl(b_i + \sum_{k\in N(i)} P_{ki}\mu_{ki}\Bigr),
    \qquad
    x_i \approx \mu_i, \quad (A^{-1})_{ii} \approx 1/P_i .
    $$

    Note the sign: $P_{ij} = -A_{ij}^2 / P_{i\setminus j}$ is **negative**. Messages are not probability distributions. Rather, they are *information updates*, and what a neighbour tells you here is "you are less certain than you thought": each message pushes a node's belief from the conditional variance $1/A_{ii}$ towards the marginal variance $(A^{-1})_{ii} \ge 1/A_{ii}$.

    Everything on the right-hand side is indexed by $i$ and its neighbours. There is no $n$ anywhere in the update.
    """)
    return


@app.cell
def _(np, sp):
    def edge_list(A):
        """Directed off-diagonal edges of a structurally symmetric sparse matrix.

        Returns (src, dst, a, rev): a = A[src, dst], and rev[e] is the index of the
        reverse edge of e — the only bookkeeping the 'exclude what j told me' rule needs."""
        Ao = sp.coo_matrix(A - sp.diags(A.diagonal()))
        src, dst, a = Ao.row, Ao.col, Ao.data
        pos = {(int(i), int(j)): e for e, (i, j) in enumerate(zip(src, dst))}
        rev = np.array([pos[(int(j), int(i))] for i, j in zip(src, dst)], dtype=int)
        return src, dst, a, rev

    return (edge_list,)


@app.cell
def _(edge_list, np, sp):
    def gabp(A, b, iters=500, tol=1e-10, damping=0.0, schedule="parallel",
             jacobi=False, record=False):
        """Gaussian belief propagation for A x = b  (Shental et al. 2008, Algorithms 1–2).

        Messages live on directed edges and carry two scalars: a precision P and a
        precision-weighted mean W = P·μ.  Every update touches one node and its
        neighbours only.

        schedule : 'parallel' — flooding; all nodes send simultaneously (à la Jacobi)
                   'serial'   — sweep nodes, using messages as soon as they arrive (à la Gauss–Seidel)
        jacobi   : clamp the precision messages to zero, which *is* Jacobi (Prop. 16)
        """
        A = sp.csr_matrix(A)
        n = A.shape[0]
        Pii = A.diagonal().astype(float)                 # self-factor precision
        Wii = np.asarray(b, dtype=float)                 # self-factor  P·μ  =  A_ii · (b_i/A_ii)
        src, dst, a, rev = edge_list(A)
        P = np.zeros(len(a))                             # message precisions
        W = np.zeros(len(a))                             # message precision × mean
        bnorm = np.linalg.norm(b)
        inbox = [np.flatnonzero(dst == i) for i in range(n)] if schedule == "serial" else None
        outbox = [np.flatnonzero(src == i) for i in range(n)] if schedule == "serial" else None

        res, mus, sds = [], [], []
        for _ in range(iters):
            if schedule == "parallel":
                SP = Pii + np.bincount(dst, weights=P, minlength=n)
                SW = Wii + np.bincount(dst, weights=W, minlength=n)
                P_ex = SP[src] - (0.0 if jacobi else P[rev])       # exclude what j told i
                W_ex = SW[src] - (0.0 if jacobi else W[rev])
                P_new = np.zeros_like(P) if jacobi else -a ** 2 / P_ex
                W_new = -a * (W_ex / P_ex)
                P = (1 - damping) * P_new + damping * P
                W = (1 - damping) * W_new + damping * W
            else:
                for i in range(n):
                    SP = Pii[i] + P[inbox[i]].sum()
                    SW = Wii[i] + W[inbox[i]].sum()
                    e = outbox[i]
                    P_ex = SP - (0.0 if jacobi else P[rev[e]])
                    W_ex = SW - (0.0 if jacobi else W[rev[e]])
                    P_new = np.zeros(len(e)) if jacobi else -a[e] ** 2 / P_ex
                    W_new = -a[e] * (W_ex / P_ex)
                    P[e] = (1 - damping) * P_new + damping * P[e]
                    W[e] = (1 - damping) * W_new + damping * W[e]

            SP = Pii + np.bincount(dst, weights=P, minlength=n)
            SW = Wii + np.bincount(dst, weights=W, minlength=n)
            mu = SW / SP
            r = np.linalg.norm(A @ mu - b) / bnorm
            res.append(r)
            if record:
                mus.append(mu.copy())
                sds.append(np.sqrt(np.abs(1.0 / SP)))
            if not np.isfinite(r) or r > 1e10:
                return dict(mu=mu, var=1.0 / SP, res=res, mus=mus, sds=sds,
                            iters=len(res), converged=False)
            if r < tol:
                break
        return dict(mu=mu, var=1.0 / SP, res=res, mus=mus, sds=sds,
                    iters=len(res), converged=res[-1] < tol)

    return (gabp,)


@app.cell
def _(mo):
    mo.md(r"""
    That is the whole solver: 30 lines, no factorisation, no inner products, no $n$-dimensional linear algebra. `np.bincount` is standing in for what would be, on real hardware, each node summing its own inbox.

    ### 4.3 Sanity check on a $3\times3$ system

    The toy example from Shental et al. (their eq. 47) is symmetric but **indefinite**, so "the Gaussian" is not a probability distribution at all. The algebra does not care.
    """)
    return


@app.cell
def _(gabp, mo, np, sp):
    A_toy = np.array([[1.0, -2.0, 3.0], [-2.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
    b_toy = np.array([-6.0, 0.0, 2.0])
    _r = gabp(sp.csr_matrix(A_toy), b_toy, iters=200)
    _exact = np.linalg.solve(A_toy, b_toy)
    mo.md(
        f"""
    | | $x_1$ | $x_2$ | $x_3$ |
    |:--|--:|--:|--:|
    | GaBP after {_r['iters']} rounds | {_r['mu'][0]:.6f} | {_r['mu'][1]:.6f} | {_r['mu'][2]:.6f} |
    | `np.linalg.solve` | {_exact[0]:.6f} | {_exact[1]:.6f} | {_exact[2]:.6f} |

    Eigenvalues of $A$: {', '.join(f'{v:.2f}' for v in np.linalg.eigvalsh(A_toy))} — not positive definite, and yet exact.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 Trees: message passing *is* Gaussian elimination

    If the graph of $A$ has no cycles, belief propagation is exact after at most as many rounds as the diameter of the tree. In practice, this happens sooner: the messages stop changing once information has crossed a correlation length, not the whole graph. Shental et al. (Prop. 14) makes the correspondence precise: on a tree, the message sweep from the leaves inward performs exactly the row operations of Gaussian elimination ($P_{i\setminus j}$ is the updated pivot $A_{ii} - \sum_l A_{li}^2/A_{ll}$), and reading off the marginals is forward substitution.

    A tridiagonal system is the simplest instance: GaBP on a chain **is** the Thomas algorithm, re-derived as inference. Below, both the solution and the marginal variances $(A^{-1})_{ii}$ come out to machine precision and the variances are the diagonal of a dense inverse whose explicit formation was avoided.
    """)
    return


@app.cell
def _(chain_matrix, gabp, np):
    n_chain = 80
    A_chain = chain_matrix(n_chain, diag=2.5)
    _rng = np.random.default_rng(2026)
    b_chain = _rng.standard_normal(n_chain)
    chain_bp = gabp(A_chain, b_chain, iters=2000, tol=1e-13)
    _Ad = A_chain.toarray()
    chain_x = np.linalg.solve(_Ad, b_chain)
    chain_v = np.diag(np.linalg.inv(_Ad))
    return A_chain, b_chain, chain_bp, chain_v, chain_x, n_chain


@app.cell
def _(
    PAL,
    base_layout,
    chain_bp,
    chain_v,
    chain_x,
    go,
    hex_rgba,
    mo,
    n_chain,
    np,
):
    _sd = np.sqrt(chain_bp["var"])
    _t = np.arange(n_chain)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.concatenate([_t, _t[::-1]]),
                              y=np.concatenate([chain_bp["mu"] + 2 * _sd, (chain_bp["mu"] - 2 * _sd)[::-1]]),
                              fill="toself", fillcolor=hex_rgba(PAL["blue"], 0.15),
                              line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="belief ± 2σ"))
    _fig.add_trace(go.Scatter(x=_t, y=chain_bp["mu"], mode="lines", name="GaBP marginal means",
                              line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=_t, y=chain_x, mode="lines", name="exact solution",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    base_layout(_fig, title=f"Chain of {n_chain} unknowns — converged in {chain_bp['iters']} rounds",
                xlabel="node i", ylabel="xᵢ", legend=dict(x=0.01, y=1.14, orientation="h"))
    mo.vstack([
        _fig,
        mo.md(
            f"""
    | | max abs. error |
    |:--|--:|
    | means vs `np.linalg.solve` | {np.max(np.abs(chain_bp['mu'] - chain_x)):.2e} |
    | variances vs `diag(inv(A))` | {np.max(np.abs(chain_bp['var'] - chain_v)):.2e} |
    """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.5 What "belief" means before convergence

    Run the message passing for $k$ rounds and stop. What is node $i$ holding?

    The marginal of the **computation tree of depth $k$** rooted at $i$. This is the graph you get by unrolling the neighbourhood of $i$ for $k$ hops. That is the sub-problem whose information has physically reached node $i$ in $k$ rounds of communication. So the belief at iteration $k$ is not a heuristic error estimate; it is the *exact posterior of the part of the problem the node has seen so far*, and the sequence interpolates from

    $$
    \text{iteration } 0:\quad \mathcal{N}\bigl(b_i/A_{ii},\; 1/A_{ii}\bigr)
    \qquad\text{(the conditional: "my equation, neighbours assumed known")}
    $$

    to

    $$
    \text{convergence}:\quad \mathcal{N}\bigl((A^{-1}b)_i,\; (A^{-1})_{ii}\bigr)
    \qquad\text{(the marginal: the whole system accounted for).}
    $$

    Each node's uncertainty therefore *grows* as information arrives. The early over-confidence of "I'll just solve my own row" is corrected by neighbours. The belief is **local and anytime**: every node has one at every round, computed from the messages it happens to hold, with no global quantity ever assembled.

    In the story of §1: at round 0 every tile reports the temperature it would reach if it were the only warm thing on the die, and reports it with the confidence of the isolated. Round by round it learns that its neighbours are warm, that theirs are, and that it sits in a hot region of the chip — its estimate rises and its stated certainty falls. After $k$ rounds a tile has accounted for exactly the $k$-hop patch of silicon around it. **That patch is what its belief describes** — not, as the next box insists, how wrong its number is.

    Watch the front of information sweep across the lattice.
    """)
    return


@app.cell
def _(A_grid, b_grid, gabp, np):
    # Same 24 × 24 test problem as §2; here it is run through the message-passing solver.
    grid_bp = gabp(A_grid, b_grid, iters=400, tol=1e-12, record=True)
    var_grid = np.diag(np.linalg.inv(A_grid.toarray()))     # reference marginals (dense, n = 576)
    return grid_bp, var_grid


@app.cell
def _(grid_bp, mo):
    round_slider = mo.ui.slider(1, min(60, grid_bp["iters"]), step=1, value=3,
                                label="message-passing rounds k", full_width=True)
    round_slider
    return (round_slider,)


@app.cell
def _(go, grid_bp, m_grid, np, round_slider, var_grid, x_grid):
    _k = round_slider.value - 1
    _mu = grid_bp["mus"][_k].reshape(m_grid, m_grid)
    _sd = grid_bp["sds"][_k].reshape(m_grid, m_grid)
    _sd_true = np.sqrt(var_grid).reshape(m_grid, m_grid)

    _fig = go.Figure()
    _fig.add_trace(go.Heatmap(z=_mu, colorscale="RdBu", zmid=0, colorbar=dict(x=0.44, len=0.9),
                              zmin=float(x_grid.min()), zmax=float(x_grid.max())))
    _fig.add_trace(go.Heatmap(z=_sd, colorscale="Viridis", xaxis="x2", yaxis="y2",
                              colorbar=dict(x=1.0, len=0.9),
                              zmin=float(_sd_true.min()) * 0.95, zmax=float(_sd_true.max()) * 1.02))
    _fig.update_layout(
        template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40),
        title=(f"round k = {round_slider.value}:  belief mean μᵢ (left) and belief std √(1/Pᵢ) (right)"),
        xaxis=dict(domain=[0.0, 0.42], visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, autorange="reversed"),
        xaxis2=dict(domain=[0.56, 0.98], visible=False),
        yaxis2=dict(visible=False, anchor="x2", scaleanchor="x2", scaleratio=1, autorange="reversed"),
    )
    _fig
    return


@app.cell
def _(grid_bp, mo, np, round_slider, var_grid, x_grid):
    _k = round_slider.value - 1
    _mu, _sd = grid_bp["mus"][_k], grid_bp["sds"][_k]
    _sd_true = np.sqrt(var_grid)
    mo.md(
        f"""
    | after k = {round_slider.value} rounds | value |
    |:--|--:|
    | relative residual ‖Aμ − b‖/‖b‖ | {grid_bp['res'][_k]:.2e} |
    | max error in the means | {np.max(np.abs(_mu - x_grid)):.2e} |
    | mean belief std (BP) | {_sd.mean():.4f} |
    | mean marginal std (exact) | {_sd_true.mean():.4f} |
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **What to look for.** At $k=1$ every node reports $b_i/A_{ii}$ and a uniformly small standard deviation: maximal over-confidence. As rounds pass, the mean fills in from the sources outward, and the standard-deviation map inflates from the boundary inward, because nodes near the boundary genuinely *are* better determined (Dirichlet conditions pin them) while interior nodes must wait to learn how loosely they are held. Both fields stop changing once the information has travelled a correlation length.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.6 Loops: exact means, over-confident variances

    On a graph with cycles the same information arrives at a node by several routes and gets double-counted. The remarkable fact (Weiss & Freeman 2001) is that this does **not** spoil the means: *if* GaBP converges, the marginal means are the exact solution $A^{-1}b$, cycles or no cycles. The variances are another matter — the computation tree that BP effectively solves keeps re-entering the same loop, and the walk-sum analysis of Malioutov, Johnson & Willsky (2006) shows BP counts only the self-return walks that revisit the root once. On a model where all those walks contribute with the same sign, the missing terms are positive, so BP **under-estimates** the variance: the solver is over-confident.
    """)
    return


@app.cell
def _(PAL, base_layout, go, grid_bp, mo, np, var_grid, x_grid):
    _v_bp, _v_true = grid_bp["var"], var_grid
    _lo, _hi = float(min(_v_bp.min(), _v_true.min())), float(max(_v_bp.max(), _v_true.max()))
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=[_lo, _hi], y=[_lo, _hi], mode="lines", name="exact",
                              line=dict(color=PAL["black"], width=1.5, dash="dash")))
    _fig.add_trace(go.Scatter(x=_v_true, y=_v_bp, mode="markers", name="one node",
                              marker=dict(color=PAL["blue"], size=6, opacity=0.55,
                                          line=dict(color=PAL["white"], width=0.5))))
    base_layout(_fig, title="Converged GaBP variances vs the true diagonal of A⁻¹",
                xlabel="(A⁻¹)ᵢᵢ  (exact)", ylabel="1/Pᵢ  (belief propagation)",
                legend=dict(x=0.02, y=0.98))
    _fig.update_layout(height=400)
    mo.vstack([
        _fig,
        mo.md(
            f"""
    | converged GaBP on the 24×24 lattice | |
    |:--|--:|
    | max error in the **means** | {np.max(np.abs(grid_bp['mu'] - x_grid)):.2e} |
    | ratio BP variance / true variance | {(_v_bp / _v_true).min():.3f} – {(_v_bp / _v_true).max():.3f} |
    """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.7 Comparison to Jacobi method

    Take the algorithm above and make two changes:

    1. clamp every precision message to zero, $P_{ij} := 0$;
    2. stop excluding the reverse message — let node $i$ use what $j$ told it when replying to $j$.

    What remains is $\mu_i = A_{ii}^{-1}\bigl(b_i - \sum_{k \neq i} A_{ki}\mu_k\bigr)$, i.e. the Jacobi iteration (Shental et al., Prop. 16).

    The classical method is not an alternative to the probabilistic one; it is the probabilistic one, marginalised down to a point estimate. Unlike Jacobi, GaBP carries precisions and excludes the reverse message. It is just bookkeeping about information.
    """)
    return


@app.cell
def _(A_grid, b_grid, gabp, mo, np, stationary):
    _K = 40
    _clamped = gabp(A_grid, b_grid, iters=_K, jacobi=True, tol=0.0, record=True)
    _jac = stationary(A_grid, b_grid, _K + 1, kind="jacobi")   # x⁰ = 0, so xᵏ⁺² is round k of BP
    _gap = max(np.max(np.abs(_m - _x)) for _m, _x in zip(_clamped["mus"], _jac[2:]))
    _full = gabp(A_grid, b_grid, iters=_K, tol=0.0)
    _bn = np.linalg.norm(b_grid)
    mo.md(
        f"""
    | | |
    |:--|--:|
    | max difference over {_K} rounds between "GaBP with $P_{{ij}} := 0$" and Jacobi | **{_gap:.2e}** |
    | relative residual after {_K} rounds — Jacobi | {np.linalg.norm(A_grid @ _jac[_K] - b_grid) / _bn:.2e} |
    | relative residual after {_K} rounds — full GaBP | {_full['res'][-1]:.2e} |

    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.8 Scheduling

    * **Synchronous (flooding)** — every node sends every round, using the previous round's messages.
    * **Asynchronous (serial)** — sweep the nodes and use each message the moment it exists.

    Neither needs an inner product, a norm, or any other quantity that couples all $n$ unknowns. Convergence is not destroyed by nodes running at different speeds, by stale messages, or by a node dropping out for a while. This is what makes the scheme viable on an unreliable, heterogeneous, or genuinely geographically distributed machine.
    """)
    return


@app.cell
def _(
    A_chain,
    A_grid,
    b_chain,
    b_grid,
    conjugate_gradients,
    gabp,
    np,
    stationary,
):
    def residual_curve(A, b, xs):
        _bn = np.linalg.norm(b)
        return [np.linalg.norm(A @ x - b) / _bn for x in xs]

    K_cmp = 120
    curves_grid = {
        "Jacobi": residual_curve(A_grid, b_grid, stationary(A_grid, b_grid, K_cmp, "jacobi")),
        "Gauss–Seidel": residual_curve(A_grid, b_grid, stationary(A_grid, b_grid, K_cmp, "gs")),
        "conjugate gradients": residual_curve(A_grid, b_grid, conjugate_gradients(A_grid, b_grid, K_cmp)),
        "GaBP (flooding)": [1.0] + gabp(A_grid, b_grid, iters=K_cmp, tol=1e-14)["res"],
        "GaBP (serial)": [1.0] + gabp(A_grid, b_grid, iters=K_cmp, tol=1e-14, schedule="serial")["res"],
    }
    curves_chain = {
        "Jacobi": residual_curve(A_chain, b_chain, stationary(A_chain, b_chain, K_cmp, "jacobi")),
        "conjugate gradients": residual_curve(A_chain, b_chain, conjugate_gradients(A_chain, b_chain, K_cmp)),
        "GaBP (flooding)": [1.0] + gabp(A_chain, b_chain, iters=K_cmp, tol=1e-14)["res"],
    }
    return curves_chain, curves_grid


@app.cell
def _(mo):
    problem_pick = mo.ui.radio(options=["24×24 lattice (loopy)", "chain of 80 (tree)"],
                               value="24×24 lattice (loopy)", label="problem", inline=True)
    problem_pick
    return (problem_pick,)


@app.cell
def _(PAL, base_layout, curves_chain, curves_grid, go, np, problem_pick):
    _curves = curves_grid if problem_pick.value.startswith("24") else curves_chain
    _style = {
        "Jacobi": (PAL["gray"], "dot"),
        "Gauss–Seidel": (PAL["pink"], "dot"),
        "conjugate gradients": (PAL["orange"], "dashdot"),
        "GaBP (flooding)": (PAL["blue"], "solid"),
        "GaBP (serial)": (PAL["green"], "solid"),
    }
    _fig = go.Figure()
    for _name, _c in _curves.items():
        _col, _dash = _style[_name]
        _fig.add_trace(go.Scatter(x=np.arange(len(_c)), y=np.maximum(_c, 1e-16), mode="lines",
                                  name=_name, line=dict(color=_col, width=2, dash=_dash)))
    base_layout(_fig, title="Relative residual ‖Ax − b‖ / ‖b‖ per iteration",
                xlabel="iteration", ylabel="relative residual", legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(type="log", range=[-14, 0.5])
    _fig.update_layout(height=430)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.9 Does it scale?

    Per round, each node sends one two-scalar message per incident edge: the cost is $O(\mathrm{nnz})$ arithmetic and $O(\mathrm{nnz})$ communication, all of it nearest-neighbour, all of it parallel. So the only question that matters is how the round count grows with $n$.

    The parameter $c$ sets the correlation length $\ell \sim 1/\sqrt{c}$ of the Gaussian field $\mathcal{N}(A^{-1}b, A^{-1})$. A node's marginal is determined by the nodes within a few $\ell$ of it; everything beyond is screened off. Information therefore has to travel a *fixed physical distance*, not across the whole domain. So the round count actually saturates. It stops growing with $n$, and the total work is $O(n)$ with perfect parallelism.

    At $c = 0$ the correlation length is the domain size, and every node needs to hear from every other. This is not a defect of message passing; it is the same long-range coupling that makes unpreconditioned Jacobi and CG slow.
    """)
    return


@app.cell
def _(bump_forcing, gabp, grid_matrix, np):
    scale_sizes = [8, 12, 16, 24, 32, 48, 64]
    scale_screens = [0.0, 0.4, 2.0]
    scale_data = {}
    for _c in scale_screens:
        _row = []
        for _m in scale_sizes:
            _A = grid_matrix(_m, screening=_c)
            _r = gabp(_A, bump_forcing(_m), iters=40000, tol=1e-8)
            _row.append(dict(n=_m * _m, m=_m, iters=_r["iters"], nnz=int(_A.nnz),
                             work=_r["iters"] * int(_A.nnz)))
        scale_data[_c] = _row
    scale_np = {c: np.array([[d["n"], d["iters"], d["work"]] for d in r]) for c, r in scale_data.items()}
    return scale_data, scale_np, scale_screens


@app.cell
def _(PAL, base_layout, go, scale_np, scale_screens):
    _fig = go.Figure()
    for _c, _col in zip(scale_screens, [PAL["orange"], PAL["blue"], PAL["green"]]):
        _d = scale_np[_c]
        _fig.add_trace(go.Scatter(x=_d[:, 0], y=_d[:, 1], mode="lines+markers",
                                  name=f"c = {_c}   (ℓ ≈ {'∞' if _c == 0 else round(1/_c**0.5, 1)})",
                                  line=dict(color=_col, width=2),
                                  marker=dict(size=7, line=dict(color=PAL["white"], width=1))))
    base_layout(_fig, title="Rounds to ‖Aμ − b‖/‖b‖ < 10⁻⁸, five-point stencil (c − Δ)",
                xlabel="number of unknowns n", ylabel="message-passing rounds",
                legend=dict(x=0.02, y=0.98))
    _fig.update_xaxes(type="log")
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=430)
    _fig
    return


@app.cell
def _(mo, scale_data, scale_screens):
    _hdr = "| n | " + " | ".join(f"rounds (c = {c})" for c in scale_screens) + " |"
    _sep = "|---|" + "---|" * len(scale_screens)
    _rows = []
    for _k in range(len(scale_data[scale_screens[0]])):
        _n = scale_data[scale_screens[0]][_k]["n"]
        _rows.append(f"| {_n} | " + " | ".join(str(scale_data[c][_k]["iters"]) for c in scale_screens) + " |")
    mo.md("\n".join([_hdr, _sep] + _rows) +
          "\n\nAcross a 64-fold increase in $n$, the screened columns grow by a factor of two or less "
          "while the unscreened one grows with the diameter of the domain. "
          "A column that flattens is an $O(n)$ solver with no global communication.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.10 When it fails

    GaBP is not unconditionally convergent. The sufficient conditions are:

    * **tree** structure on $A$,
    * Strict diagonal dominance in $A$ (Weiss & Freeman 2001),
    * **walk-summability**, i.e., $\rho\bigl(|I - D^{-1}A|\bigr) < 1$ for $D = \operatorname{diag}(A)$ (Malioutov et al. 2006).

    In practice, GaBP will often converge outside these conditions as well. But it does have an edge. Below, a lattice with random $\pm w$ couplings (a "frustrated" model, the sort where loops carry conflicting information) sweeps from harmless to divergent. The walk-summability bound is crossed long before anything goes wrong, and then convergence fails somewhere near the point where the Gaussian stops being a valid distribution at all.
    """)
    return


@app.cell
def _(grid_matrix, np, sp):
    def frustrated_matrix(m, w, seed=3):
        "Lattice with random ±w couplings and unit diagonal — loops with conflicting information."
        rng = np.random.default_rng(seed)
        G = sp.coo_matrix(grid_matrix(m) - sp.diags(grid_matrix(m).diagonal()))
        sign, rows, cols, vals = {}, [], [], []
        for i, j in zip(G.row, G.col):
            key = (min(int(i), int(j)), max(int(i), int(j)))
            if key not in sign:
                sign[key] = rng.choice([-1.0, 1.0])
            rows.append(i); cols.append(j); vals.append(w * sign[key])
        Ao = sp.coo_matrix((vals, (rows, cols)), shape=(m * m, m * m))
        return (sp.eye(m * m) + Ao).tocsr()

    return (frustrated_matrix,)


@app.cell
def _(mo):
    coupling = mo.ui.slider(0.05, 0.35, step=0.01, value=0.2, label="coupling strength w", full_width=True)
    damping_ui = mo.ui.slider(0.0, 0.9, step=0.1, value=0.0, label="damping", full_width=True)
    mo.vstack([coupling, damping_ui])
    return coupling, damping_ui


@app.cell
def _(coupling, damping_ui, frustrated_matrix, gabp, np):
    m_fr = 12
    A_fr = frustrated_matrix(m_fr, coupling.value)
    _rng = np.random.default_rng(7)
    b_fr = _rng.standard_normal(m_fr * m_fr)
    fr_run = gabp(A_fr, b_fr, iters=600, tol=1e-10, damping=damping_ui.value)
    _Ad = A_fr.toarray()
    _D = np.diag(1.0 / np.diag(_Ad))
    fr_diag = dict(
        rho=float(np.max(np.abs(np.linalg.eigvals(np.abs(np.eye(m_fr * m_fr) - _D @ _Ad))))),
        lam=float(np.linalg.eigvalsh(_Ad).min()),
        dd=float(np.min(np.abs(np.diag(_Ad)) - (np.abs(_Ad).sum(1) - np.abs(np.diag(_Ad))))),
    )
    return fr_diag, fr_run


@app.cell
def _(PAL, base_layout, fr_diag, fr_run, go, mo, np):
    _res = np.array(fr_run["res"])
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=np.arange(1, len(_res) + 1), y=np.clip(np.nan_to_num(_res, nan=1e10), 1e-16, 1e10),
                              mode="lines", name="GaBP",
                              line=dict(color=PAL["blue"] if fr_run["converged"] else PAL["orange"], width=2)))
    base_layout(_fig, title="relative residual" + ("" if fr_run["converged"] else "  —  DIVERGED"),
                xlabel="round", ylabel="‖Aμ − b‖ / ‖b‖", showlegend=False)
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=340)
    mo.vstack([
        _fig,
        mo.md(
            f"""
    | diagnostic | value | verdict |
    |:--|--:|:--|
    | diagonal dominance margin $\\min_i \\bigl(\\lvert A_{{ii}}\\rvert - \\sum_{{j\\neq i}}\\lvert A_{{ij}}\\rvert\\bigr)$ | {fr_diag['dd']:+.3f} | {'dominant' if fr_diag['dd'] > 0 else 'not dominant'} |
    | walk-summability $\\rho\\bigl(\\lvert I - D^{{-1}}A\\rvert\\bigr)$ | {fr_diag['rho']:.3f} | {'walk-summable' if fr_diag['rho'] < 1 else 'not walk-summable'} |
    | smallest eigenvalue $\\lambda_{{\\min}}(A)$ | {fr_diag['lam']:+.3f} | {'valid Gaussian' if fr_diag['lam'] > 0 else 'not a distribution'} |
    | GaBP | {fr_run['iters']} rounds | {'converged' if fr_run['converged'] else 'diverged'} |
    """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    Try it. Push $w$ up from 0.05. The unit diagonal is beaten by the four couplings at $w = 0.25$ and walk-summability goes one step later at $w \approx 0.26$. The two sufficient conditions fail together, and neither failure costs anything: the solver keeps converging, taking 45 rounds at $w = 0.26$ and 297 at $w = 0.29$. It breaks between $w = 0.29$ and $w = 0.30$, which is essentially where $A$ stops being positive definite ($\lambda_{\min} = +0.007$ at $w = 0.30$). Then turn on damping, $P \leftarrow (1-\alpha)P_{\text{new}} + \alpha P_{\text{old}}$: it buys smoothness in the borderline regime but does *not* rescue the indefinite case. Sharp characterisations of the convergence basin, and principled fixes outside it, remain open (see Johnson et al. 2009, Ruozzi & Tatikonda 2013).
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feedback

    Found a bug, a claim that does not hold up, or something we could explain better? Please open an
    issue at [github.com/biaslab/ProbNum2026-Tutorial](https://github.com/biaslab/ProbNum2026-Tutorial/issues).
    """)
    return


if __name__ == "__main__":
    app.run()
