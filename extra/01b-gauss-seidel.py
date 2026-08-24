# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""The Gauss-Seidel method.

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
    # The Gauss–Seidel method

    **ProbNum 2026 tutorial — companion to notebook 01**

    Jacobi (notebook 01a) has every tile update using its neighbours' values *from the previous round*. Half of those values are already stale: by the time tile 17 updates, tiles 1 through 16 have new numbers sitting in memory, and Jacobi deliberately ignores them.

    Gauss–Seidel is the one-word fix: **use whatever has arrived**. Sweep the tiles in some order and update each one in place,

    $$
    x_i \;\leftarrow\; \frac{1}{4+c}\Bigl(b_i + \sum_{j\sim i} x_j\Bigr) \qquad\text{with the current contents of } x .
    $$

    The formula is identical to Jacobi's. The only difference is that $x$ on the right is being overwritten as we go, so a tile that updates late in the sweep sees fresh values from its earlier neighbours.

    This buys two things and costs one:

    * it roughly **halves the iteration count** (§2), and needs no second copy of $x$;
    * it makes the method **depend on the ordering** (§3) — which turns out to be a feature, and is the same design freedom that notebook 01 calls *message scheduling*;
    * it makes the sweep **sequential**, which on a parallel machine is a real problem — until §4 fixes it.
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
def _(np, sp):
    def sor(A, b, iters, omega=1.0, order=None):
        """Successive over-relaxation; omega = 1 is exactly Gauss-Seidel.

        The sweep is written as an explicit loop over nodes, because that is what the
        method *is* — every other formulation hides the in-place update."""
        A = sp.csr_matrix(A)
        n = A.shape[0]
        indptr, indices, data = A.indptr, A.indices, A.data
        d = A.diagonal()
        order = np.arange(n) if order is None else np.asarray(order)
        x = np.zeros(n)
        out = [x.copy()]
        for _ in range(iters):
            for i in order:
                s = 0.0
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    if j != i:
                        s += data[k] * x[j]          # x[j] may already be this sweep's value
                x[i] = (1 - omega) * x[i] + omega * (b[i] - s) / d[i]
            out.append(x.copy())
        return out

    def jacobi(A, b, iters, omega=1.0):
        "Damped Jacobi, for comparison (notebook 01a)."
        A = sp.csr_matrix(A)
        d = A.diagonal()
        x = np.zeros_like(b, dtype=float)
        out = [x.copy()]
        for _ in range(iters):
            x = x + omega * (b - A @ x) / d
            out.append(x.copy())
        return out
    return jacobi, sor


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. The splitting, and the iteration matrix

    Split $A = L + D + U$ into its strictly lower, diagonal and strictly upper parts — *in the ordering you intend to sweep*. Jacobi solved with $D$; Gauss–Seidel solves with everything already available when the update happens, which is $D + L$:

    $$
    (D+L)\,x^{(t+1)} = b - U x^{(t)},
    \qquad
    M_{\mathrm{GS}} = -(D+L)^{-1}U .
    $$

    That looks like it needs a triangular solve, but it does not: sweeping the nodes in order and updating in place *is* the forward substitution. The cost is the same as a Jacobi step.

    More generally, take the Gauss–Seidel update and move only a fraction $\omega$ of the way — or, for $\omega > 1$, overshoot. This is **successive over-relaxation (SOR)**:

    $$
    x_i \leftarrow (1-\omega)\,x_i + \omega \cdot (\text{Gauss–Seidel value}),
    \qquad
    M_{\mathrm{SOR}}(\omega) = (D + \omega L)^{-1}\bigl((1-\omega)D - \omega U\bigr).
    $$

    $\omega = 1$ is Gauss–Seidel. The surprise of §5 is how much a well-chosen $\omega \approx 1.4$ is worth.
    """
    )
    return


@app.cell
def _(bump_forcing, grid_matrix, np):
    m_grid = 24
    c_grid = 0.4
    A_grid = grid_matrix(m_grid, screening=c_grid)
    b_grid = bump_forcing(m_grid)
    Ad_grid = A_grid.toarray()
    x_star = np.linalg.solve(Ad_grid, b_grid)

    def iteration_matrices(Ad, omega=1.0, perm=None):
        "M_Jacobi and M_SOR for a dense A, optionally reordered by perm."
        if perm is not None:
            Ad = Ad[np.ix_(perm, perm)]
        D = np.diag(np.diag(Ad))
        L = np.tril(Ad, -1)
        U = np.triu(Ad, 1)
        M_j = -np.linalg.solve(D, L + U)
        M_s = np.linalg.solve(D + omega * L, (1 - omega) * D - omega * U)
        return M_j, M_s

    def rho(M):
        return float(np.abs(np.linalg.eigvals(M)).max())
    return (
        A_grid,
        Ad_grid,
        b_grid,
        c_grid,
        iteration_matrices,
        m_grid,
        rho,
        x_star,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Twice as fast, and why exactly twice

    For a class of matrices called **consistently ordered** — which includes the five-point stencil swept in the natural row-by-row order — there is an exact relationship between the two spectral radii:

    $$
    \boxed{\;\rho(M_{\mathrm{GS}}) \;=\; \rho(M_{\mathrm{J}})^2 .\;}
    $$

    Each Gauss–Seidel sweep does what two Jacobi sweeps do. Not "about twice as good" — exactly, in the asymptotic rate. It is worth being clear about what this does *not* say: it halves the iteration count, but the count is still governed by $\rho_{\mathrm J}^2 \to 1$ as the grid refines. Gauss–Seidel is a constant factor, not a change of asymptotics. Both are slow for the same reason, and §5's SOR is the classical way out.
    """
    )
    return


@app.cell
def _(Ad_grid, iteration_matrices, mo, rho):
    _M_j, _M_gs = iteration_matrices(Ad_grid, omega=1.0)
    rho_j = rho(_M_j)
    rho_gs = rho(_M_gs)

    mo.md(
        f"""
    | | value |
    |:--|--:|
    | $\\rho(M_\\mathrm{{J}})$ | {rho_j:.6f} |
    | $\\rho(M_\\mathrm{{J}})^2$ | {rho_j ** 2:.6f} |
    | $\\rho(M_\\mathrm{{GS}})$ | {rho_gs:.6f} |
    | ratio | {rho_gs / rho_j ** 2:.9f} |

    Exact to nine digits, on a 576-unknown problem nobody tuned.
    """
    )
    return (rho_j,)


@app.cell
def _(A_grid, PAL, b_grid, base_layout, go, jacobi, np, sor, x_star):
    _K = 160
    _curves = {}
    for _name, _xs in (("Jacobi", jacobi(A_grid, b_grid, _K)),
                       ("Gauss–Seidel", sor(A_grid, b_grid, _K, omega=1.0))):
        _curves[_name] = [np.linalg.norm(x - x_star) / np.linalg.norm(x_star) for x in _xs]

    _fig = go.Figure()
    for _name, _col in (("Jacobi", PAL["gray"]), ("Gauss–Seidel", PAL["blue"])):
        _fig.add_trace(go.Scatter(x=np.arange(len(_curves[_name])),
                                  y=np.maximum(_curves[_name], 1e-16),
                                  mode="lines", name=_name,
                                  line=dict(color=_col, width=2)))
    base_layout(_fig, title="Relative error ‖x − x⋆‖ / ‖x⋆‖ per sweep",
                xlabel="sweep", ylabel="relative error",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
    _fig.update_yaxes(type="log", range=[-13, 0.5])
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. The ordering is a real choice

    Jacobi has no ordering — every tile uses last round's values, so the sweep order cannot matter. Gauss–Seidel has one, and it changes the iteration matrix: $L$ and $U$ are defined *relative to the sweep order*, so permuting the unknowns gives a genuinely different method.

    Three orderings of the same lattice:

    * **natural** — row by row, left to right. Information travels one row per sweep in the sweep direction and instantly along it.
    * **reverse** — the same sweep backwards. Different iteration matrix, same spectral radius here, but information now flows the other way.
    * **red–black** — colour the lattice like a checkerboard. Every red tile has only black neighbours and vice versa, so a sweep splits into *two* half-sweeps, and within each half-sweep **nothing depends on anything else**.

    Red–black is the interesting one, and §4 is about why.
    """
    )
    return


@app.cell
def _(PAL, base_layout, go, np):
    def red_black_order(m):
        "Indices of the m×m lattice, all red (i+j even) first, then all black."
        idx = np.arange(m * m)
        r, c = idx // m, idx % m
        red = idx[(r + c) % 2 == 0]
        black = idx[(r + c) % 2 == 1]
        return np.concatenate([red, black]), red, black

    def colour_figure(m=8):
        _, _red, _black = red_black_order(m)
        _fig = go.Figure()
        for _grp, _col, _name in ((_red, PAL["orange"], "red  (updated first)"),
                                  (_black, PAL["black"], "black  (updated second)")):
            _fig.add_trace(go.Scatter(x=_grp % m, y=-(_grp // m), mode="markers",
                                      name=_name,
                                      marker=dict(color=_col, size=16,
                                                  line=dict(color=PAL["white"], width=1.5))))
        base_layout(_fig, title=f"Red–black colouring of an {m}×{m} lattice",
                    legend=dict(x=0.01, y=1.15, orientation="h"))
        _fig.update_xaxes(visible=False)
        _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        _fig.update_layout(height=380)
        return _fig
    return colour_figure, red_black_order


@app.cell
def _(colour_figure):
    colour_figure(8)
    return


@app.cell
def _(Ad_grid, iteration_matrices, m_grid, mo, np, red_black_order, rho, rho_j):
    _perm_rb, _, _ = red_black_order(m_grid)
    _perm_rev = np.arange(m_grid ** 2)[::-1]

    _rows = []
    for _name, _perm in (("natural (row by row)", None),
                         ("reverse", _perm_rev),
                         ("red–black", _perm_rb)):
        _, _M = iteration_matrices(Ad_grid, omega=1.0, perm=_perm)
        _rows.append(f"| {_name} | {rho(_M):.6f} | {rho(_M) / rho_j ** 2:.6f} |")

    mo.md(
        "| ordering | ρ(M_GS) | ÷ ρ(M_J)² |\n|:--|--:|--:|\n" + "\n".join(_rows) +
        """

All three are consistently ordered, so all three land on exactly $\\rho_\\mathrm{J}^2$. The rate is
the same; what differs is **how much of the sweep can happen at once**.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Red–black: Gauss–Seidel's rate at Jacobi's parallelism

    In the natural ordering, tile $i$ needs the *new* value of tile $i-1$. That is a chain of dependencies $n$ long: the sweep is inherently sequential, and on a parallel machine Gauss–Seidel's factor-of-two advantage evaporates because you cannot run it.

    Red–black removes the dependency without changing the rate. Every red tile's neighbours are black and every black tile's neighbours are red, so:

    1. update **all** red tiles simultaneously — each needs only black values, none of which change during this half-sweep;
    2. exchange;
    3. update **all** black tiles simultaneously — each now sees the new red values.

    Two halo exchanges per sweep instead of one, full parallelism within each half, and the spectral radius is still $\rho_\mathrm{J}^2$. This is the standard way Gauss–Seidel is actually run on a parallel machine, and as a multigrid smoother.

    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **This is a message schedule.** The two half-sweeps are precisely a **two-colour message schedule**. Notebook 01 §4.8 compares a *flooding* schedule (all nodes send at once, like Jacobi) with a *serial* one (nodes use messages as they arrive, like Gauss–Seidel) and measures 112 rounds against 67 — the same factor, for the same reason. Red–black is how you get the serial schedule's speed without the serial schedule's dependency chain, and it applies verbatim to belief propagation.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(A_grid, b_grid, m_grid, mo, np, red_black_order, sor, x_star):
    _K = 120
    _perm_rb, _, _ = red_black_order(m_grid)
    _nat = sor(A_grid, b_grid, _K, omega=1.0)
    _rb = sor(A_grid, b_grid, _K, omega=1.0, order=_perm_rb)

    def _hit(xs, tol=1e-10):
        e = [np.linalg.norm(x - x_star) / np.linalg.norm(x_star) for x in xs]
        return next((i for i, v in enumerate(e) if v < tol), None)

    mo.md(
        f"""
    | ordering | sweeps to relative error 10⁻¹⁰ | parallel work per sweep |
    |:--|--:|:--|
    | natural | {_hit(_nat)} | 1 sequential chain of length n |
    | red–black | {_hit(_rb)} | 2 fully parallel half-sweeps |

    The asymptotic rates are identical, so the small difference in sweep count is transient, not
    structural. Same method, same speed, completely different machine.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Over-relaxation, and the one place classical theory is spectacular

    Gauss–Seidel moves each tile to the value its own equation currently asks for. Look at a sequence of such updates and you notice they keep moving in the *same direction* — the method is consistently under-shooting. So overshoot on purpose: take $\omega > 1$.

    For a consistently ordered matrix with $\rho_\mathrm{J} = \rho$, the optimal relaxation factor and the rate it achieves are known in closed form:

    $$
    \omega_\ast = \frac{2}{1 + \sqrt{1 - \rho^2}},
    \qquad
    \rho\bigl(M_{\mathrm{SOR}}(\omega_\ast)\bigr) = \omega_\ast - 1 .
    $$

    This is not a small effect. Below, $\rho_\mathrm{J} \approx 0.90$ and $\rho_\mathrm{GS} \approx 0.81$ — both hopeless — while $\rho_{\mathrm{SOR}}(\omega_\ast) \approx 0.40$, which is a completely different solver. The catch is the one every classical method has: $\omega_\ast$ depends on $\rho_\mathrm{J}$, which you do not know, and the curve is **sharply asymmetric** — undershooting $\omega_\ast$ costs little, overshooting it degrades fast. Move the slider onto the cliff and watch.
    """
    )
    return


@app.cell
def _(mo):
    omega_ui = mo.ui.slider(1.0, 1.95, step=0.01, value=1.0,
                            label="relaxation ω", full_width=True)
    omega_ui
    return (omega_ui,)


@app.cell
def _(Ad_grid, PAL, base_layout, go, iteration_matrices, np, omega_ui, rho, rho_j):
    _ws = np.arange(1.0, 1.96, 0.025)
    _rhos = [rho(iteration_matrices(Ad_grid, omega=w)[1]) for w in _ws]
    _w_star = 2 / (1 + np.sqrt(1 - rho_j ** 2))

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_ws, y=_rhos, mode="lines", name="ρ(M_SOR(ω))",
                              line=dict(color=PAL["blue"], width=2)))
    _fig.add_trace(go.Scatter(x=[_w_star], y=[_w_star - 1], mode="markers",
                              name="theory: ω⋆, ω⋆ − 1",
                              marker=dict(color=PAL["orange"], size=13, symbol="star",
                                          line=dict(color=PAL["white"], width=1))))
    _fig.add_trace(go.Scatter(x=[omega_ui.value],
                              y=[rho(iteration_matrices(Ad_grid, omega=omega_ui.value)[1])],
                              mode="markers", name="slider",
                              marker=dict(color=PAL["black"], size=10)))
    base_layout(_fig, title="Spectral radius of SOR against the relaxation factor",
                xlabel="ω", ylabel="ρ", legend=dict(x=0.02, y=0.98))
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(Ad_grid, iteration_matrices, mo, np, omega_ui, rho, rho_j):
    _w = omega_ui.value
    _w_star = 2 / (1 + np.sqrt(1 - rho_j ** 2))
    _rho_w = rho(iteration_matrices(Ad_grid, omega=_w)[1])
    _rho_star = rho(iteration_matrices(Ad_grid, omega=_w_star)[1])

    mo.md(
        f"""
    | | value |
    |:--|--:|
    | ρ(M_J) | {rho_j:.6f} |
    | ρ(M_GS) = ρ(M_J)² | {rho_j ** 2:.6f} |
    | theoretical ω⋆ = 2/(1 + √(1 − ρ_J²)) | {_w_star:.6f} |
    | theoretical rate ω⋆ − 1 | {_w_star - 1:.6f} |
    | measured ρ at ω⋆ | {_rho_star:.6f} |
    | **ρ at your ω = {_w:.2f}** | **{_rho_w:.6f}** |
    | sweeps to gain 8 digits at this ω | {int(np.ceil(np.log(1e-8) / np.log(_rho_w))) if _rho_w < 1 else '∞'} |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. What to take to notebook 01

    Gauss–Seidel contributes two ideas to the message-passing story, and one warning.

    **The idea that survives.** *Use information as soon as it exists.* Nothing about that requires a matrix splitting, a triangular solve or even an ordering — it is a statement about scheduling, and it transfers directly to belief propagation, where notebook 01 measures the same factor of roughly two (67 sweeps against 112). Red–black shows that the scheduling freedom and parallelism are not in conflict.

    **The idea that also survives.** *Relaxation.* Notebook 01's GaBP has a `damping` parameter doing the arithmetic that $\omega < 1$ does here, for the same purpose: stabilising an iteration that would otherwise oscillate. In the classical setting the optimal parameter has a closed form; in the loopy-graph setting it does not, and finding it is still open.

    **The warning.** Everything quantitative on this page — $\rho_{\mathrm{GS}} = \rho_\mathrm{J}^2$, the closed form for $\omega_\ast$ — depends on *consistent ordering*, a property of this particular matrix and sweep. Change the geometry and the theory does not simply degrade; it stops applying. The message-passing view does not fix that, but it does make the assumption visible: consistent ordering is a statement about how information flows through the graph, which is exactly what a schedule is.

    ## Exercises

    1. **Break consistent ordering.** Sweep the lattice in a random order and measure $\rho(M_{\mathrm{GS}})$. Is it still $\rho_\mathrm{J}^2$? Try ten random orders and plot the spread.
    2. **Symmetric Gauss–Seidel.** Do a forward sweep and then a backward one. Show the resulting iteration is symmetric positive definite (so it is usable as a CG preconditioner, notebook 01c) and compare its rate with plain Gauss–Seidel.
    3. **The SOR cliff.** Measure the actual sweep count to $10^{-10}$ for $\omega \in \{1.0, 1.2, \omega_\ast, 1.6, 1.9\}$. How much does overshooting cost compared with undershooting by the same amount?
    4. **ω⋆ without knowing ρ.** Estimate $\rho_\mathrm{J}$ from the observed ratio of successive residual norms during the first 20 Jacobi sweeps, feed it into the formula, and see how close the resulting $\omega$ gets you.
    5. **Red–black on a different graph.** A checkerboard colouring works because the lattice is bipartite. Find a graph where it is not, and think about what the analogous schedule would be. (This is graph colouring, and it is how parallel Gauss–Seidel is done in general.)

    ## References

    * Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. Chapter 4.
    * Young, D. M. (1971). *Iterative Solution of Large Linear Systems*. Academic Press. The consistent-ordering and SOR theory.
    * Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*, 2nd ed. SIAM. Red–black smoothing.
    """
    )
    return


if __name__ == "__main__":
    app.run()
