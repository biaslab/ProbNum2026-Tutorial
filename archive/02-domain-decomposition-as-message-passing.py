# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""Domain decomposition as message passing — and what the belief does not know.

ProbNum 2026 tutorial, notebook 2: parallel PDE solvers as Gaussian belief propagation.
"""
import marimo

__generated_with = "0.17.6"
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

    def hex_rgba(hexcolor, alpha):
        h = hexcolor.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return PAL, base_layout, hex_rgba


@app.cell
def _(mo):
    mo.md(
        r"""
    # Domain decomposition as message passing — and what the belief does not know

    **ProbNum 2026 tutorial — notebook 2**

    Notebook 1 established the reframing: solving $Ax = b$ is marginal inference in a Gaussian Markov
    random field, and Jacobi is Gaussian belief propagation with the second moment deleted. It ended
    honestly — on one machine, conjugate gradients wins on iterations.

    This notebook puts the same construction where parallel PDE solvers actually live: a domain
    decomposed across ranks. Three things follow, and the third is the one that matters for
    probabilistic numerics.

    1. **Domain decomposition methods *are* message passing.** The message from subdomain $i$ to
       subdomain $j$ is exactly a **Schur complement** — the discrete Dirichlet-to-Neumann map of $i$
       seen from $j$. On a chain of subdomains, belief propagation *is* block substructuring. This is
       not an analogy; we check it to machine zero.
    2. **The decomposition granularity is a dial** running continuously from a direct solver (one
       subdomain) to fully local belief propagation (one node per subdomain). Coarser blocks buy fewer
       iterations *and* less uncertainty bias, at more work per rank. That is the cluster-variation idea,
       arrived at from the numerical-analysis side.
    3. **The belief is blind to its own numerical error.** The covariance $A^{-1}$ is a property of the
       *operator*, not of the computation — provably so, because the precision recursion never sees $b$.
       Meanwhile the question a PDE solver actually needs answered — *when should I stop iterating?* — is
       decided by the discretisation error, which the belief also knows nothing about.

    Point 3 is the probabilistic-numerics content. Points 1 and 2 are what make it a statement about a
    method people really run.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. The problem and the three solutions

    Solve the screened Poisson (reaction–diffusion) equation on the unit square with Dirichlet
    conditions,

    $$
    (c - \Delta)\, u = f \quad \text{on } (0,1)^2, \qquad u|_{\partial\Omega} = 0,
    $$

    discretised with the five-point stencil on an $m \times m$ grid, in the usual $h^2$-scaled form so
    that $A$ has diagonal $4 + ch^2$ and off-diagonals $-1$. We use the **manufactured solution**

    $$
    u(x,y) = \sin(\pi x)\,\sin(\pi y)\,e^{\,x+y},
    $$

    from which $f = (c - \Delta)u$ follows analytically, so the exact answer is known.

    The exponential factor is not decoration. A bare $\sin \pi x \sin \pi y$ is an *exact eigenvector* of
    the five-point stencil, so CG would converge in a single iteration and the discretisation error would
    be atypically clean — a rigged comparison. A pure polynomial such as $x(1-x)y(1-y)$ is worse: the
    stencil is exact for cubics, so there would be no discretisation error to measure at all. This
    choice vanishes on the boundary, stays smooth, and has broad spectral content.

    That gives us three distinct objects, and keeping them apart is the whole point of §6–7:

    | symbol | what it is | error from the previous one |
    |:--|:--|:--|
    | $u$ | the true solution of the PDE | — |
    | $u_h = A^{-1}b$ | the exact solution of the *discrete* system | **discretisation error** |
    | $\mu^{(k)}$ | what the solver has after $k$ rounds | **algebraic error** |

    A probabilistic numerical method should have something to say about both gaps. We will find that
    this one says nothing about either.
    """
    )
    return


@app.cell
def _(np, sp):
    def poisson(m, screening=0.0):
        "Five-point stencil for (c − Δ) on an m×m grid, Dirichlet BC, scaled by h²."
        d = 4.0 + screening
        T = sp.diags([-np.ones(m - 1), d * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
        band = sp.diags([-np.ones(m - 1), -np.ones(m - 1)], [-1, 1])
        return (sp.kron(sp.eye(m), T) + sp.kron(band, sp.eye(m))).tocsr()

    def manufactured(m, c_phys):
        """u = sin(πx) sin(πy) e^{x+y} for (c − Δ)u = f, on the grid, with the h²-scaled RHS.

        The exponential modulation matters: a bare sin·sin is an exact eigenvector of the
        five-point stencil, which would make both the discretisation error and CG's iteration
        count unrepresentative (CG would converge in one step). A pure polynomial is worse
        still — the stencil is exact for cubics, so there would be no discretisation error at
        all. This choice vanishes on the boundary, is smooth, and has broad spectral content."""
        h = 1.0 / (m + 1)
        g = np.arange(1, m + 1) * h
        X, Y = np.meshgrid(g, g, indexing="ij")
        pi, E = np.pi, np.exp(X + Y)
        sx, sy, cx, cy = np.sin(pi * X), np.sin(pi * Y), np.cos(pi * X), np.cos(pi * Y)
        u = sx * sy * E
        lap = E * (2 * (1 - pi ** 2) * sx * sy + 2 * pi * (cx * sy + sx * cy))
        f = c_phys * u - lap
        return u.ravel(), (h ** 2) * f.ravel(), h

    def partition(m, px, py):
        "Cartesian decomposition of the grid into px×py subdomains; returns labels and index sets."
        r = np.repeat(np.arange(m), m)
        c = np.tile(np.arange(m), m)
        lab = np.minimum(r * px // m, px - 1) * py + np.minimum(c * py // m, py - 1)
        return lab, [np.flatnonzero(lab == k) for k in range(px * py)]

    def block_system(A, idx):
        "Dense diagonal blocks A_ii and the couplings A_ij between adjacent subdomains."
        P = len(idx)
        Ad = A.toarray()
        Aii = [Ad[np.ix_(I, I)] for I in idx]
        Aij, nbr = {}, [[] for _ in range(P)]
        for i in range(P):
            for j in range(P):
                if i != j:
                    B = Ad[np.ix_(idx[i], idx[j])]
                    if np.any(B):
                        Aij[(i, j)] = B
                        nbr[i].append(j)
        return Aii, Aij, nbr
    return block_system, manufactured, partition, poisson


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. The decomposition is a coarser factor graph

    Partition the grid into $P$ subdomains, one per rank. Collect the unknowns of subdomain $i$ into a
    single vector-valued variable $x_i$. Then $A$ becomes **block sparse**: $A_{ij} \neq 0$ only when
    subdomains $i$ and $j$ share an interface, and the non-zeros of $A_{ij}$ sit only on the interface
    nodes.

    So the factor graph of notebook 1 — one node per unknown — coarsens into a factor graph with **one
    node per rank**, whose edges are exactly the halo exchanges a parallel code already performs. Nothing
    about the mathematics changed; we regrouped the variables.
    """
    )
    return


@app.cell
def _(PAL, base_layout, go, np, partition):
    _m, _px, _py = 32, 4, 4
    _lab, _idx = partition(_m, _px, _py)
    _fig = go.Figure()
    _fig.add_trace(go.Heatmap(z=_lab.reshape(_m, _m), colorscale="Portland", showscale=False,
                              hoverinfo="skip", opacity=0.55))
    _s = _m // _px
    for _k in range(1, _px):
        _fig.add_hline(y=_k * _s - 0.5, line=dict(color=PAL["black"], width=2))
    for _k in range(1, _py):
        _fig.add_vline(x=_k * (_m // _py) - 0.5, line=dict(color=PAL["black"], width=2))
    base_layout(_fig, title=f"{_m}×{_m} grid decomposed into {_px}×{_py} subdomains "
                            f"({(_m//_px)*(_m//_py)} unknowns per rank)")
    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    _fig.update_layout(height=470)
    _fig
    return


@app.cell
def _(np):
    def block_gabp(Aii, Aij, nbr, bloc, rounds=400, tol=1e-11, x_ref=None,
                   wake=None, rng=None, record=False, msg0=None):
        """Block Gaussian BP over subdomains, in information form.

        The message subdomain i sends to j is

            Λ_{i→j} = −A_ij^T (Λ_{i∖j})^{-1} A_ij,     η_{i→j} = −A_ij^T (Λ_{i∖j})^{-1} η_{i∖j},

        which — see §3 — is precisely a Schur complement on their shared interface.
        `wake` gives a per-rank probability of updating in a round (load imbalance)."""
        P = len(Aii)
        if msg0 is None:
            Lam = {(i, j): np.zeros((len(bloc[j]),) * 2) for i in range(P) for j in nbr[i]}
            Eta = {(i, j): np.zeros(len(bloc[j])) for i in range(P) for j in nbr[i]}
        else:
            Lam = {k: v[0].copy() for k, v in msg0.items()}
            Eta = {k: v[1].copy() for k, v in msg0.items()}

        hist, mus = [], []
        for _ in range(rounds):
            act = range(P) if wake is None else [i for i in range(P) if rng.random() < wake[i]]
            new = {}
            for i in act:
                Li = Aii[i] + sum(Lam[(k, i)] for k in nbr[i])
                ei = bloc[i] + sum(Eta[(k, i)] for k in nbr[i])
                for j in nbr[i]:
                    B = Aij[(i, j)]
                    S = np.linalg.solve(Li - Lam[(j, i)],
                                        np.column_stack([B, ei - Eta[(j, i)]]))
                    new[(i, j)] = (-B.T @ S[:, :-1], -B.T @ S[:, -1])
            for k, v in new.items():
                Lam[k], Eta[k] = v
            mu = np.concatenate([
                np.linalg.solve(Aii[i] + sum(Lam[(k, i)] for k in nbr[i]),
                                bloc[i] + sum(Eta[(k, i)] for k in nbr[i])) for i in range(P)])
            if record:
                mus.append(mu.copy())
            if x_ref is not None:
                hist.append(float(np.abs(mu - x_ref).max()))
                if hist[-1] < tol:
                    break
        cov = [np.linalg.inv(Aii[i] + sum(Lam[(k, i)] for k in nbr[i])) for i in range(P)]
        return dict(mu=mu, cov=cov, hist=hist, mus=mus, rounds=len(hist) if hist else rounds,
                    msg={k: (Lam[k], Eta[k]) for k in Lam})
    return (block_gabp,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. The message is a Schur complement

    Take the simplest case: two subdomains, so

    $$
    A = \begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11}\end{pmatrix}.
    $$

    Subdomain 0 has no other neighbour, so $\Lambda_{0\setminus 1} = A_{00}$ and its single message is

    $$
    \Lambda_{0\to1} = -A_{01}^\top A_{00}^{-1} A_{01} = -A_{10}A_{00}^{-1}A_{01}.
    $$

    Subdomain 1's belief precision is therefore

    $$
    \Lambda_1 = A_{11} + \Lambda_{0\to1} = A_{11} - A_{10}A_{00}^{-1}A_{01} \;=\; S,
    $$

    **the Schur complement of $A_{00}$ in $A$** — and $\eta_1$ is the correspondingly eliminated
    right-hand side. Solving $\Lambda_1 \mu_1 = \eta_1$ is block Gaussian elimination.

    In domain-decomposition language, $-\Lambda_{0\to1}$ restricted to the shared interface *is* the
    discrete **Dirichlet-to-Neumann map** of subdomain 0: everything subdomain 1 needs to know about the
    physics on the other side of the interface, and nothing else. Its rank equals the number of interface
    nodes, not the number of unknowns in the subdomain — which is why the messages are cheap.
    """
    )
    return


@app.cell
def _(block_system, mo, np, partition, poisson):
    _m = 16
    _A = poisson(_m, 0.5)
    _lab, _idx = partition(_m, 2, 1)
    _Aii, _Aij, _nbr = block_system(_A, _idx)
    _msg = -_Aij[(0, 1)].T @ np.linalg.solve(_Aii[0], _Aij[(0, 1)])
    _S = _Aii[1] - _Aij[(1, 0)] @ np.linalg.solve(_Aii[0], _Aij[(0, 1)])
    schur_gap = float(np.abs((_Aii[1] + _msg) - _S).max())
    schur_rank = int(np.linalg.matrix_rank(_msg, tol=1e-9))
    schur_dofs = len(_idx[0])
    mo.md(
        f"""
    | two subdomains of a {_m}×{_m} grid | |
    |:--|--:|
    | $\\lVert (A_{{11}} + \\Lambda_{{0\\to1}}) - S \\rVert_\\infty$ | **{schur_gap:.2e}** |
    | unknowns per subdomain | {schur_dofs} |
    | rank of the message $\\Lambda_{{0\\to1}}$ | **{schur_rank}** (= interface nodes) |

    Exactly zero — not approximately. The message *is* the Schur complement, and it is a rank-{schur_rank}
    object living on the interface even though the subdomain carries {schur_dofs} unknowns.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Extend this to a **chain** of subdomains — a 1-D strip decomposition. The subdomain graph is a tree,
    so belief propagation is exact (notebook 1, §4.4), and the sweep of messages from both ends performs
    exactly the sequence of Schur complements that block substructuring performs. It terminates in $P-1$
    rounds, the diameter of the chain:
    """
    )
    return


@app.cell
def _(block_gabp, block_system, manufactured, mo, np, partition, poisson, spla):
    _m = 24
    _A = poisson(_m, 0.5)
    _u, _b, _h = manufactured(_m, 0.5 * (_m + 1) ** 2)
    chain_rows = []
    for _P in [2, 3, 4, 6, 8, 12]:
        _lab, _idx = partition(_m, _P, 1)
        _Aii, _Aij, _nbr = block_system(_A, _idx)
        _perm = np.concatenate(_idx)
        _xr = spla.spsolve(_A.tocsc(), _b)[_perm]
        _r = block_gabp(_Aii, _Aij, _nbr, [_b[I] for I in _idx],
                        rounds=60, tol=1e-11, x_ref=_xr)
        chain_rows.append((_P, _r["rounds"], _r["hist"][-1]))
    mo.md(
        "| subdomains in the chain | rounds to converge | final error |\n|---:|---:|---:|\n"
        + "\n".join(f"| {p} | **{r}** | {e:.1e} |" for p, r, e in chain_rows)
        + "\n\nExactly $P-1$ rounds, every time: message passing on a chain of subdomains is the block "
          "Thomas algorithm. Substructuring was always message passing; nobody wrote it that way."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Granularity is a dial between a direct solver and belief propagation

    Now vary the decomposition on a fixed problem. One subdomain means one block, one exact solve — a
    **direct solver**. One node per subdomain is the scalar belief propagation of notebook 1. Everything
    in between is a domain-decomposition method, and the same code runs all of them.

    Three quantities move together as blocks get smaller: iterations go **up**, work per rank goes
    **down**, and the bias in the belief's variance goes **up**. That last one is not a coincidence —
    each block accounts *exactly* for every loop inside it, so coarsening the blocks absorbs loops that
    scalar BP has to approximate. This is the cluster-variation / generalised-BP correction (the second
    algorithm in Fanaskov 2022), reached from the numerical-analysis direction.
    """
    )
    return


@app.cell
def _(block_gabp, block_system, manufactured, np, partition, poisson, spla):
    GRAN_M = 32
    _c_phys = 800.0
    _h = 1.0 / (GRAN_M + 1)
    A_gran = poisson(GRAN_M, _c_phys * _h ** 2)
    _u, b_gran, _ = manufactured(GRAN_M, _c_phys)
    _Sig = np.linalg.inv(A_gran.toarray())

    gran = []
    for _px, _py in [(1, 1), (2, 1), (2, 2), (4, 4), (8, 8), (16, 16)]:
        _lab, _idx = partition(GRAN_M, _px, _py)
        _Aii, _Aij, _nbr = block_system(A_gran, _idx)
        _perm = np.concatenate(_idx)
        _xr = spla.spsolve(A_gran.tocsc(), b_gran)[_perm]
        _r = block_gabp(_Aii, _Aij, _nbr, [b_gran[I] for I in _idx],
                        rounds=400, tol=1e-11, x_ref=_xr)
        _vb = np.concatenate([np.diag(c) for c in _r["cov"]])
        _vt = np.diag(_Sig)[_perm]
        gran.append(dict(px=_px, py=_py, P=_px * _py, dofs=len(_idx[0]),
                         rounds=_r["rounds"], ratio_lo=float((_vb / _vt).min()),
                         ratio_hi=float((_vb / _vt).max())))
    return GRAN_M, gran


@app.cell
def _(PAL, base_layout, go, gran):
    _fig = go.Figure()
    _P = [g["P"] for g in gran]
    _fig.add_trace(go.Scatter(x=_P, y=[g["rounds"] for g in gran], mode="lines+markers",
                              name="rounds to converge", line=dict(color=PAL["blue"], width=2),
                              marker=dict(size=9)))
    _fig.add_trace(go.Scatter(x=_P, y=[1 - g["ratio_lo"] for g in gran], mode="lines+markers",
                              name="worst variance bias  1 − BP/true", yaxis="y2",
                              line=dict(color=PAL["orange"], width=2, dash="dashdot"),
                              marker=dict(size=9, symbol="diamond")))
    base_layout(_fig, title="Decomposition granularity: direct solver (left) → scalar BP (right)",
                xlabel="number of subdomains P", ylabel="rounds",
                legend=dict(x=0.02, y=0.98))
    _fig.update_xaxes(type="log")
    _fig.update_layout(height=430, yaxis2=dict(title="variance bias", overlaying="y",
                                               side="right", showgrid=False))
    _fig
    return


@app.cell
def _(GRAN_M, mo, gran):
    _rows = "\n".join(
        f"| {g['px']}×{g['py']} | {g['P']} | {g['dofs']} | {g['rounds']} | "
        f"{g['ratio_lo']:.4f} – {g['ratio_hi']:.4f} |" for g in gran)
    mo.md(
        f"""
    | decomposition | ranks | unknowns/rank | rounds | variance ratio BP / true |
    |---:|---:|---:|---:|---:|
    {_rows}

    On a {GRAN_M}×{GRAN_M} grid. The top row is a direct solve: one round, and the variances are
    **exact**. As blocks shrink, rounds grow and the belief becomes progressively over-confident. The
    two ends of this table are usually taught in different courses.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. The parallel cost model

    On $P$ ranks the currency is not flops but **synchronisation**. Take one nearest-neighbour halo
    exchange as the unit of latency:

    * **A GaBP round costs 1.** Each rank sends its interface messages to its cartesian neighbours. That
      is the halo exchange a stencil code already does, carrying a Schur complement instead of a
      boundary strip.
    * **A CG iteration costs $1 + \log_2 P$.** One halo exchange for the matrix–vector product, plus a
      global **all-reduce** for the two inner products — a tree reduction over all ranks, and a hard
      barrier. This is precisely why communication-avoiding and pipelined Krylov methods exist.

    The all-reduce is also where **load imbalance** bites. Adaptive meshes, heterogeneous hardware and OS
    jitter make ranks run at different speeds; a barrier means every rank waits for the slowest. Message
    passing has no barrier, so a slow rank delays only the information that must flow *through* it.
    """
    )
    return


@app.cell
def _(np):
    def cg_iters(A, b, x_ref, tol, iters=4000, x0=None):
        "Plain CG, returning the iteration at which the max-norm error first drops below tol."
        x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
        r = b - A @ x
        p = r.copy()
        rr = r @ r
        for it in range(1, iters + 1):
            Ap = A @ p
            al = rr / (p @ Ap)
            x = x + al * p
            r = r - al * Ap
            rn = r @ r
            p = r + (rn / rr) * p
            rr = rn
            if np.abs(x - x_ref).max() < tol:
                return it
        return None
    return (cg_iters,)


@app.cell
def _(block_system, manufactured, np, partition, poisson, spla):
    M_MAIN, PX, PY, C_PHYS = 64, 8, 8, 800.0
    P_MAIN = PX * PY
    H_MAIN = 1.0 / (M_MAIN + 1)
    A_main = poisson(M_MAIN, C_PHYS * H_MAIN ** 2)
    u_exact, b_main, _ = manufactured(M_MAIN, C_PHYS)
    lab_main, idx_main = partition(M_MAIN, PX, PY)
    Aii_m, Aij_m, nbr_m = block_system(A_main, idx_main)
    perm_main = np.concatenate(idx_main)
    u_h = spla.spsolve(A_main.tocsc(), b_main)[perm_main]
    bloc_main = [b_main[I] for I in idx_main]
    DISC_ERR = float(np.abs(u_h - u_exact[perm_main]).max())
    SOLVE_TOL = 1e-10 * float(np.abs(u_h).max())
    return (A_main, Aii_m, Aij_m, C_PHYS, DISC_ERR, H_MAIN, M_MAIN, P_MAIN, PX,
            PY, SOLVE_TOL, b_main, bloc_main, idx_main, nbr_m, perm_main, u_exact, u_h)


@app.cell
def _(Aii_m, Aij_m, SOLVE_TOL, block_gabp, bloc_main, nbr_m, u_h):
    bp_main = block_gabp(Aii_m, Aij_m, nbr_m, bloc_main, rounds=600, tol=SOLVE_TOL,
                         x_ref=u_h, record=True)
    return (bp_main,)


@app.cell
def _(A_main, SOLVE_TOL, b_main, cg_iters, perm_main, spla):
    # CG works in the original ordering; compare against the same solution vector.
    _x_nat = spla.spsolve(A_main.tocsc(), b_main)
    K_CG = cg_iters(A_main, b_main, _x_nat, SOLVE_TOL, iters=4000)
    return (K_CG,)


@app.cell
def _(DISC_ERR, K_CG, M_MAIN, P_MAIN, bp_main, mo, np):
    _allred = np.log2(P_MAIN)
    _gabp = bp_main["rounds"]
    _cg = K_CG * (1 + _allred)
    mo.md(
        f"""
    | {M_MAIN}×{M_MAIN} grid, {P_MAIN} ranks, all-reduce = $\\log_2 P$ = {_allred:.0f} halo latencies | |
    |:--|--:|
    | GaBP rounds × 1 halo exchange | **{_gabp}** latency units |
    | CG {K_CG} iterations × (1 halo + {_allred:.0f} all-reduce) | **{_cg:.0f}** latency units |
    | ratio | **{_cg/_gabp:.1f}×** |
    | discretisation error $\\lVert u_h - u\\rVert_\\infty$ | {DISC_ERR:.2e} |

    Note what this is *not* saying: CG still needs fewer iterations. It is the price of each iteration
    that differs, and that price is set by the machine, not by the mathematics.
    """
    )
    return


@app.cell
def _(Aii_m, Aij_m, K_CG, P_MAIN, SOLVE_TOL, block_gabp, bloc_main, nbr_m, np, u_h):
    strag = []
    for _S in [1, 2, 5, 10, 25, 50]:
        _wake = np.ones(P_MAIN)
        _wake[P_MAIN // 2 + 3] = 1.0 / _S           # one straggler rank, S times slower
        _r = block_gabp(Aii_m, Aij_m, nbr_m, bloc_main, rounds=3000, tol=SOLVE_TOL,
                        x_ref=u_h, wake=_wake, rng=np.random.default_rng(1))
        strag.append(dict(S=_S, rounds=_r["rounds"],
                          cg=K_CG * (_S + np.log2(P_MAIN))))
    return (strag,)


@app.cell
def _(PAL, base_layout, go, mo, strag):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=[s["S"] for s in strag], y=[s["rounds"] for s in strag],
                              mode="lines+markers", name="GaBP (no barrier)",
                              line=dict(color=PAL["blue"], width=2), marker=dict(size=9)))
    _fig.add_trace(go.Scatter(x=[s["S"] for s in strag], y=[s["cg"] for s in strag],
                              mode="lines+markers", name="CG (barrier every iteration)",
                              line=dict(color=PAL["orange"], width=2, dash="dashdot"),
                              marker=dict(size=9, symbol="diamond")))
    base_layout(_fig, title="One straggler rank: time to solution",
                xlabel="straggler slowdown factor S", ylabel="latency units",
                legend=dict(x=0.02, y=0.98))
    _fig.update_xaxes(type="log")
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=400)
    mo.vstack([
        _fig,
        mo.md(
            "| S | GaBP | CG | ratio |\n|---:|---:|---:|---:|\n"
            + "\n".join(f"| {s['S']} | {s['rounds']} | {s['cg']:.0f} | {s['cg']/s['rounds']:.1f}× |"
                        for s in strag)
            + "\n\n**Be honest about this one.** GaBP is not immune to a straggler — information that must "
              "pass *through* the slow rank is still delayed, and the round count grows roughly linearly "
              "in $S$ as well. What it avoids is the barrier: the other 63 ranks keep working. The "
              "advantage is a solid constant factor, not a change of asymptotics."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. What is the covariance, when there is no noise?

    Now the probabilistic-numerics question, which this setting poses much more sharply than any
    estimation problem does. In notebook 1 the Gaussian was

    $$
    p(x) \;\propto\; \exp\bigl(-\tfrac12 x^\top A x + b^\top x\bigr) \;=\; \mathcal{N}\bigl(A^{-1}b,\ A^{-1}\bigr),
    $$

    and in a statistical problem — a GP posterior, a sensor network — the covariance $A^{-1}$ is
    inherited from measurement noise and everyone knows what it means.

    **Here there is no noise.** $A$ is a discretised differential operator. $b$ is a forcing. Nothing was
    measured. So what is $A^{-1}$?

    It is the **discrete Green's function** of $(c - \Delta)$. And that is a genuinely meaningful object:
    by the SPDE representation of Lindgren, Rue & Lindström (2011), a Gaussian field with precision
    $(c - \Delta)$ *is* a Matérn field, so $\mathrm{diag}(A^{-1})$ is (up to scaling) the pointwise
    variance of the Matérn field whose correlation length is $\ell = 1/\sqrt{c}$. The belief is the prior
    that a probabilistic PDE solver would have chosen anyway — it is handed to us by the operator.

    That is a satisfying answer, and it contains the problem. **$A^{-1}$ is a property of the operator.
    It is the same object whether you solve the system by Cholesky, by CG, or by message passing, and the
    same whether you have taken one round or a thousand.** It cannot be a statement about your
    computation, because it does not depend on your computation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. The belief is blind to its own numerical error

    This is not a calibration defect to be tuned away; it is structural. Look at the precision recursion:

    $$
    \Lambda_{i\to j} = -A_{ij}^\top \bigl(\Lambda_{i\setminus j}\bigr)^{-1} A_{ij},
    \qquad
    \Lambda_{i\setminus j} = A_{ii} + \!\!\sum_{k \in N(i)\setminus j}\!\! \Lambda_{k\to i}.
    $$

    **There is no $b$ anywhere in it.** The precisions form a closed dynamical system driven by $A$
    alone. The means ride on top of them and are the only part that ever sees the data. So the reported
    uncertainty cannot know how far the mean has got — and we can demonstrate that with no ambiguity at
    all: run the *same operator* with two different right-hand sides.
    """
    )
    return


@app.cell
def _(Aii_m, Aij_m, M_MAIN, b_main, block_gabp, idx_main, nbr_m, np, perm_main, spla, A_main):
    _rng = np.random.default_rng(4)
    _b_rough = np.zeros(M_MAIN * M_MAIN)
    _b_rough[_rng.choice(M_MAIN * M_MAIN, 12, replace=False)] = _rng.standard_normal(12)
    blind = []
    for _name, _b in [("smooth forcing (manufactured)", b_main),
                      ("rough forcing (12 point sources)", _b_rough)]:
        _xr = spla.spsolve(A_main.tocsc(), _b)[perm_main]
        _r = block_gabp(Aii_m, Aij_m, nbr_m, [_b[I] for I in idx_main],
                        rounds=12, tol=0.0, x_ref=_xr, record=True)
        _sd = np.concatenate([np.sqrt(np.diag(c)) for c in _r["cov"]])
        blind.append(dict(name=_name, errs=_r["hist"], sd=_sd,
                          rel=[e / np.abs(_xr).max() for e in _r["hist"]]))
    blind_sd_gap = float(np.abs(blind[0]["sd"] - blind[1]["sd"]).max())
    return blind, blind_sd_gap


@app.cell
def _(blind, blind_sd_gap, mo):
    _rows = "\n".join(
        f"| {k+1} | {blind[0]['rel'][k]:.3e} | {blind[1]['rel'][k]:.3e} | "
        f"{blind[1]['rel'][k]/blind[0]['rel'][k]:.1f}× |" for k in range(0, 12, 2))
    mo.md(
        f"""
    | round | relative error, smooth $b$ | relative error, rough $b$ | ratio |
    |---:|---:|---:|---:|
    {_rows}

    | | |
    |:--|--:|
    | max difference between the two **reported** standard deviations | **{blind_sd_gap:.2e}** |

    The two problems have actual errors differing by a factor of several at every round, and the solver
    reports **bit-identical** uncertainty for both. A belief that is the same for a problem you have
    nearly solved and a problem you have not is not a belief about the solving.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. The question a PDE solver actually needs answered

    Here is where the blindness costs something concrete. For a discretised PDE, iterating until the
    algebraic error is far below the discretisation error is **wasted work**: you are computing $u_h$ to
    ten digits when $u_h$ itself only agrees with $u$ to five. The classical stopping rule is therefore

    $$
    \lVert \mu^{(k)} - u_h \rVert \;\lesssim\; \lVert u_h - u \rVert,
    $$

    and the whole difficulty is that neither side is observable at run time. This is exactly the kind of
    question probabilistic numerics exists to answer — and below, the solver's own belief is a flat line
    through the entire decision.
    """
    )
    return


@app.cell
def _(DISC_ERR, PAL, base_layout, bp_main, go, hex_rgba, np, perm_main, u_exact, u_h):
    _alg = [float(np.abs(m - u_h).max()) for m in bp_main["mus"]]
    _tot = [float(np.abs(m - u_exact[perm_main]).max()) for m in bp_main["mus"]]
    _k = np.arange(1, len(_alg) + 1)
    _stop = next((i + 1 for i, a in enumerate(_alg) if a < DISC_ERR), None)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_k, y=_alg, mode="lines+markers", name="algebraic error ‖μᵏ − u_h‖∞",
                              line=dict(color=PAL["blue"], width=2), marker=dict(size=6)))
    _fig.add_trace(go.Scatter(x=_k, y=_tot, mode="lines+markers", name="total error ‖μᵏ − u‖∞",
                              line=dict(color=PAL["violet"], width=2), marker=dict(size=6)))
    _fig.add_hline(y=DISC_ERR, line=dict(color=PAL["orange"], dash="dash"),
                   annotation_text="discretisation error ‖u_h − u‖∞", annotation_position="right")
    if _stop:
        _fig.add_vrect(x0=_stop, x1=len(_alg), fillcolor=hex_rgba(PAL["gray"], 0.16),
                       line_width=0, annotation_text="wasted computation",
                       annotation_position="top left")
    base_layout(_fig, title="Where to stop — and the solver cannot tell",
                xlabel="message-passing round k", ylabel="max-norm error",
                legend=dict(x=0.62, y=0.98))
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=450)
    _fig
    return


@app.cell
def _(DISC_ERR, bp_main, mo, np, perm_main, u_exact, u_h):
    _alg = [float(np.abs(m - u_h).max()) for m in bp_main["mus"]]
    _tot = [float(np.abs(m - u_exact[perm_main]).max()) for m in bp_main["mus"]]
    _stop = next((i + 1 for i, a in enumerate(_alg) if a < DISC_ERR), len(_alg))
    _floor = min(_tot)
    _best = int(np.argmin(_tot)) + 1
    mo.md(
        f"""
    | | |
    |:--|--:|
    | discretisation error | {DISC_ERR:.3e} |
    | round at which the algebraic error drops below it | **{_stop}** |
    | round at which the total error is actually smallest | **{_best}** |
    | rounds the solver went on to run | {len(_alg)} |
    | improvement in total error after round {_best} | **none** — it flattens at {_floor:.2e} |

    Everything past round {_best} is arithmetic that changes no digit of the answer. A solver that knew
    its own numerical error would stop there and hand back roughly
    **{100*(1 - _best/len(_alg)):.0f}%** of the compute. Its belief, meanwhile, reached its final value
    around round {min(12, len(_alg))} and reports the same number at round {len(_alg)}.
    """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **The open problem, stated precisely.** After $k$ rounds each rank holds the exact posterior of its
    $k$-hop computation tree — a *different, truncated* problem. What we want instead is a belief over
    $u$ (not $u_h$) that accounts for (i) the messages that have not yet arrived, (ii) the loops that
    block BP still approximates, and (iii) the discretisation. None of the three is in $A^{-1}$.

    There are candidate tools rather than a solution. **Walk-sum analysis** (Malioutov, Johnson & Willsky
    2006) decomposes $(A^{-1})_{ij}$ into a sum over walks: walks longer than $k$ are exactly the
    truncation error after $k$ rounds, and the self-return walks that BP double-counts are exactly the
    variance bias — so both numerical terms have a series representation waiting to be bounded.
    **Message residuals** $\lVert \Lambda^{(k)}_{i\to j} - \Lambda^{(k-1)}_{i\to j}\rVert$ are local,
    free, and empirically track convergence — but they are a diagnostic, not a posterior. And the
    discretisation term is the classical business of probabilistic PDE solvers, which have so far been
    developed for the *serial* setting.

    Closing that gap would give something no current solver has: a parallel PDE solver that reports a
    calibrated belief about the true solution, locally, at every rank, and stops itself.
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Scorecard

    ### What this notebook establishes

    1. **Domain decomposition is message passing**, exactly: the message is a Schur complement, the
       subdomain graph is the factor graph, and on a chain it reduces to block substructuring in $P-1$
       rounds. Verified to machine zero, not by analogy.
    2. **One dial spans direct solvers and belief propagation.** Coarsening blocks trades work per rank
       against iteration count *and* against uncertainty bias, because a block absorbs its internal
       loops exactly. Cluster-variation methods are the coarse end of domain decomposition.
    3. **The parallel cost argument is real but modest**: a GaBP round is a halo exchange, a CG iteration
       is a halo exchange plus an all-reduce barrier. Worth a solid constant factor, more under load
       imbalance, and nothing at all if you are running on one machine.
    4. **The belief is blind to the computation** — provably, since the precision recursion is $b$-free —
       and it is blind to the discretisation too. The one question a PDE solver most needs answered
       (*when do I stop?*) is precisely the one it cannot answer.

    ### Where message passing loses

    * **Iteration count.** CG needs far fewer. If you are on one machine with a good preconditioner,
      use CG.
    * **Exact variances.** If the factorisation is affordable, **selected inversion** (Takahashi;
      Rue & Martino) gives $\mathrm{diag}(A^{-1})$ exactly and fast. GaBP does not win on variances.
    * **Optimality.** Classical domain decomposition has optimised transmission conditions, coarse-grid
      corrections and rigorous condition-number bounds. Plain GaBP has none of these; the honest claim is
      that it is the *same family of methods* written probabilistically, not a better member of it.
    * **Nonlinearity and non-symmetry.** Everything here needs $A = A^\top$ positive definite. See
      Fanaskov (2022) for the non-symmetric messages.

    ### The claim worth defending

    > Domain decomposition methods have been passing Gaussian messages since the 1980s without saying so.
    > Writing them that way costs nothing and makes the missing piece visible: these solvers carry a
    > belief, the belief is exactly the operator's Green's function, and it says nothing whatsoever about
    > the error of the computation that produced it. That gap is a probabilistic-numerics problem, and
    > it sits inside a method that thousands of people already run.

    ### Exercises

    1. **Two-level.** Add a coarse subdomain connected to every other (a coarse-grid correction). Does
       the round count become independent of $P$? What does it do to the variance bias — and is the
       coarse node a legitimate factor-graph node or a cheat?
    2. **Optimised transmission.** Classical optimised Schwarz replaces the Dirichlet-to-Neumann map with
       a cheap approximation. Truncate $\Lambda_{i\to j}$ to its leading $r$ eigenvectors and plot rounds
       versus $r$. How much of the Schur complement does a message actually need to carry?
    3. **Anisotropy.** Replace $-\Delta$ with $-(\partial_{xx} + \varepsilon\,\partial_{yy})$. Predict,
       then measure, how the best decomposition shape changes — and relate it to correlation length.
    4. **Residuals as an error estimate.** Test whether $\lVert \eta^{(k)}_{i\to j} - \eta^{(k-1)}_{i\to j}\rVert$
       predicts the algebraic error well enough to serve as a *local, distributed* stopping rule. Compare
       against the discretisation error. This is exercise-shaped but genuinely open.
    5. **Walk-sum truncation.** For a small grid, compute the walk-sum decomposition of $(A^{-1})_{ii}$
       explicitly and split it into walks of length $\le k$ and $> k$. Does the tail predict the
       round-$k$ error?

    ### References

    * Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief
      propagation solver for systems of linear equations*. IEEE ISIT.
      [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
    * Fanaskov, V. (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear
      equations*. SIAM J. Sci. Comput. 44(2), A77–A102. (Non-symmetric messages; cluster-variation
      matrix inversion; GaBP as a multigrid smoother.)
    * Malioutov, D. M., Johnson, J. K., & Willsky, A. S. (2006). *Walk-sums and belief propagation in
      Gaussian graphical models*. JMLR 7, 2031–2064.
    * Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between Gaussian fields and
      Gaussian Markov random fields: the SPDE approach*. JRSS-B 73(4), 423–498. (Why $\mathrm{diag}(A^{-1})$
      is a Matérn variance.)
    * Toselli, A., & Widlund, O. (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
    * Rue, H., & Martino, S. (2007). *Approximate Bayesian inference for hierarchical GMRF models*.
      (Selected inversion — the honest competitor for $\mathrm{diag}(A^{-1})$.)
    * Ortiz, J., Evans, T., & Davison, A. J. (2021). *A visual introduction to Gaussian belief
      propagation*. [arXiv:2107.02308](https://arxiv.org/abs/2107.02308).
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as
      Machine Learning*. Cambridge University Press.
    """
    )
    return


if __name__ == "__main__":
    app.run()
