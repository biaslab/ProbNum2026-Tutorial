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
    **ProbNum 2026 tutorial**

    Probabilistic numerics turns a computation into an inference problem. For linear solvers, a Krylov method is a Gaussian model that has seen $k$ matrix–vector products and reports a belief about $x_\ast = A^{-1}b$. It is also, at scale, expensive: the belief lives on all of $\mathbb{R}^n$, its covariance is dense, and every iteration needs global inner products.

    This tutorial takes a different route to the same destination. Instead of treating $A$ as a black box we can only probe, we read its **sparsity pattern as a graphical model**. Solving $Ax = b$ then becomes *marginal inference* in a Gaussian Markov random field, and the natural algorithm is **message passing**.

    Nothing is global. No inner products, no barriers, no dense covariance. The belief is *local and anytime*: after $k$ rounds every node holds a distribution built from exactly the information that has reached it.

    | | |
    |:--|:--|
    | **1. Problem specification** | large sparse systems, and why "large" forces "distributed" |
    | **2. Classical numerical approach** | direct factorisation, Jacobi/Gauss–Seidel, conjugate gradients |
    | **3. Probabilistic numerical approach** | $Ax=b$ as the mean of $\mathcal{N}(A^{-1}b,\, A^{-1})$ |
    | **4. Message-passing version** | Gaussian belief propagation, and what it brings  |
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **References**
    - Shental, Bickson, Siegel, Wolf & Dolev, *Gaussian belief propagation solver for systems of linear equations*, International Symposium on Information Theory, 2008; extended version [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
    - Bickson, Tock, Shental & Dolev, *Polynomial linear programming with Gaussian belief propagation*, Allerton Conference on Communication, Control, and Computing, 2008.
    - Fanaskov, *Gaussian belief propagation solvers for nonsymmetric systems of linear equations*, SIAM Journal on Scientific Computing,  2022.
    - Cockayne, Oates, Ipsen, & Girolami, *A Bayesian Conjugate-Gradient Method*, Bayesian Analysis, 2009.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Problem specification

    ### Why would anyone solve $Ax = b$?

    Take a tiled accelerator: a few thousand compute tiles on one die, each with its own power counters, its own temperature sensor, and its own ability to throttle its clock. The chip is thermally interesting. A tile that runs hot has to slow down — but so, often, do its neighbours, because heat spreads sideways through the silicon faster than any of them can react. To decide who throttles, the control loop needs the **steady-state temperature of every tile**, and it needs it refreshed faster than the die's thermal time constant.

    The physics fits in one line. Tile $i$ dissipates power $b_i$; it conducts heat to the four tiles it touches; and it loses heat to the coolant at a rate proportional to how far above ambient it sits. In steady state the three terms balance:

    $$
    \underbrace{c\,x_i}_{\text{lost to the coolant}} \;+\; \underbrace{\sum_{j \sim i}\,(x_i - x_j)}_{\text{conducted to neighbours}} \;=\; \underbrace{b_i}_{\text{dissipated on tile } i}.
    $$

    That is one equation per tile, each involving five unknowns, and it is exactly the five-point discretisation of the screened Poisson equation $(c - \Delta)u = f$. Stack the tile temperatures into $x$ and the dissipated powers into $b$ and the control loop is asking for

    $$
    A x = b, \qquad A \in \mathbb{R}^{n\times n}\ \ \text{symmetric},\ \text{sparse},\ \text{one row per tile.}
    $$

    Keep the die in mind, because three features of it are what this tutorial is really about — and they are not special to silicon. The same three hold for a power grid, a sensor network, a robot swarm and a domain-decomposed PDE; the die is just the case where they are hardest to argue with.

    * **$A$ is sparse and structured — it *is* the floorplan.** Row $i$ couples $x_i$ to the handful of tiles it physically touches, and to nothing else. The graph of the matrix is the layout of the chip. (Elsewhere it is a discretised differential operator, a Gauss–Markov model, a network of sensors — same picture.)
    * **The data is already distributed.** $b_i$ is a number tile $i$'s own counters measured, and row $i$ of $A$ is a property of tile $i$'s own package. Nothing was ever assembled anywhere. Assembling it means shipping every tile's telemetry to one place — every control period, forever.
    * **Global synchronisation is the bottleneck.** With a few thousand tiles, a barrier costs more than the arithmetic between barriers, and by the time everyone has checked in, the temperature field has moved. Krylov methods need two inner products per step: that is two barriers per step.

    So $n$ is large enough that the interesting quantity is not "how many flops" but **how the work is laid out across the machine** — and, since we are at a probabilistic-numerics meeting, what each tile is entitled to believe while the answer is still arriving.

    Our two running examples are the topologies that bracket the difficulty:

    * a **chain** — one row of tiles, a tridiagonal system, whose graph is a *tree*;
    * a **2-D lattice** — the whole die, a five-point stencil for $(c - \Delta)u = f$, whose graph is *loopy*.

    The parameter $c \ge 0$ is the screening (reaction) term — here, the strength of the coupling to the coolant. It already has a physical meaning: it sets how far a hotspot is felt. A well-cooled die screens each hot tile into a small halo; a poorly cooled one lets hotspots talk to each other across the chip. It has a probabilistic meaning too, which we will come back to: $c$ sets the **correlation length** of the associated Gaussian field, and with it everything about how far information has to travel.
    """)
    return


@app.cell
def _(np, sp):
    def chain_matrix(n, diag=2.5):
        "Tridiagonal system: the graph of A is a chain, i.e. a tree."
        off = -np.ones(n - 1)
        return sp.diags([off, diag * np.ones(n), off], [-1, 0, 1], format="csr")

    def grid_matrix(m, screening=0.0):
        "Five-point stencil for (c − Δ) on an m×m lattice: the graph is loopy."
        d = 4.0 + screening
        T = sp.diags([-np.ones(m - 1), d * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
        band = sp.diags([-np.ones(m - 1), -np.ones(m - 1)], [-1, 1])
        return (sp.kron(sp.eye(m), T) + sp.kron(band, sp.eye(m))).tocsr()

    def bump_forcing(m, centers=((0.3, 0.35), (0.7, 0.65)), width=0.12):
        "Two localised sources on the m×m lattice, flattened row-major."
        g = (np.arange(m) + 0.5) / m
        X, Y = np.meshgrid(g, g, indexing="ij")
        f = np.zeros((m, m))
        for k, (cx, cy) in enumerate(centers):
            f += (1.0 if k % 2 == 0 else -0.8) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
        return f.ravel()

    return bump_forcing, chain_matrix, grid_matrix


@app.cell
def _(mo):
    mo.md(r"""
    The picture to keep in mind for the rest of the tutorial: **the matrix *is* a graph**. Node $i$ is the unknown $x_i$; there is an edge $\{i,j\}$ whenever $A_{ij} \neq 0$. Everything the solver will do is expressible as nodes talking along those edges.
    """)
    return


@app.cell
def _(PAL, base_layout, go, grid_matrix, np, sp):
    _m = 6
    _A = grid_matrix(_m)
    _Ao = sp.coo_matrix(_A - sp.diags(_A.diagonal()))
    _xy = np.array([[i // _m, i % _m] for i in range(_m * _m)], dtype=float)

    _ex, _ey = [], []
    for _i, _j in zip(_Ao.row, _Ao.col):
        if _i < _j:
            _ex += [_xy[_i, 1], _xy[_j, 1], None]
            _ey += [_xy[_i, 0], _xy[_j, 0], None]

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", name="edges  (Aᵢⱼ ≠ 0)",
                              line=dict(color=PAL["gray"], width=1.5), hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=_xy[:, 1], y=_xy[:, 0], mode="markers+text",
                              text=[str(_k) for _k in range(_m * _m)], textposition="middle center",
                              textfont=dict(size=8, color=PAL["white"]),
                              marker=dict(color=PAL["blue"], size=20, line=dict(color=PAL["white"], width=1.5)),
                              name="unknowns  xᵢ"))
    base_layout(_fig, title=f"The graph of A: {_m}×{_m} lattice, {_A.nnz} non-zeros",
                legend=dict(x=0.01, y=1.12, orientation="h"))
    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, autorange="reversed")
    _fig.update_layout(height=420)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Classical numerical approach

    **Direct solvers.** Factorise $A = LL^\top$ and substitute. Exact in exact arithmetic, and for the 2-D lattice the factor $L$ suffers fill-in: a banded matrix with $O(n)$ non-zeros produces a factor with $O(n^{3/2})$ of them. Eliminating a node connects all of its neighbours to each other, so the graph densifies as you go.

    **Stationary iterative methods.** Split $A = D + R$ and iterate

    $$
    x^{(t+1)} = D^{-1}\bigl(b - R\,x^{(t)}\bigr) \qquad \text{(Jacobi)},
    $$

    i.e. *each unknown solves its own equation, assuming its neighbours are right*. This is already local and distributed, as node $i$ only needs $x_j$ for its neighbours. Gauss–Seidel is the same update with values used as soon as they are available (asynchronous rather than synchronous). Both are simple and both converge slowly, at a rate set by the spectral radius of the iteration matrix.

    **Krylov methods.** Conjugate gradients minimises the $A$-norm error over the Krylov space and converges in $O(\sqrt{\kappa}\,\log \varepsilon^{-1})$ iterations, which is far better. The price is that every step needs $r^\top r$ and $p^\top Ap$, two inner products over all $n$ entries. On a distributed machine every processor must wait for every other.

    All three return a vector $x_k$ plus an error bound in terms of quantities ($\kappa$, $\|x_\ast\|$) that are exactly as unknown as the solution.
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
    # 3. Probabilistic numerical approach

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

    So the solution vector *is* the mean of a Gaussian whose **precision matrix is $A$ itself** (Shental et al. 2008, Prop. 8). Solving a linear system and computing the marginal means of a Gaussian Markov random field are the same problem:

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

       Back on the die, that quantity is not a statistical abstraction either: $(A^{-1})_{ij}$ is the temperature rise at tile $i$ per unit of power dissipated at tile $j$ — the **thermal impedance** of the chip, which is what a thermal engineer would have measured. Its diagonal, the marginal variance, is how hot tile $i$ gets from its own watt *once the rest of the die has been allowed to warm up in response*. The conditional variance $1/A_{ii}$ is the same number computed with every neighbour pinned to ambient: the answer a tile would give if it believed it were the only warm thing on the chip. The gap between those two is real, physical, and — as §4 will show — is precisely what the messages carry.
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
    # 4. The message-passing version

    ## 4.1 The factor graph

    Write the density as a product of one factor per node and one per edge:

    $$
    p(x) \;\propto\; \prod_{i} \phi_i(x_i) \prod_{\{i,j\}} \psi_{ij}(x_i,x_j),
    \qquad
    \phi_i(x_i) = \exp\!\bigl(b_i x_i - \tfrac12 A_{ii} x_i^2\bigr),
    \qquad
    \psi_{ij}(x_i,x_j) = \exp(-x_i A_{ij} x_j).
    $$

    The self-factor $\phi_i$ is row $i$'s own equation — it is $\mathcal{N}(x_i;\, b_i/A_{ii},\, 1/A_{ii})$, exactly the "solve my equation ignoring the coupling" belief that Jacobi starts from. The edge factor $\psi_{ij}$ is the coupling. **Every quantity in the factor graph is an entry of $A$ or $b$ that node $i$ already owns.**

    ## 4.2 The messages

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
    ## 4.5 What "belief" means before convergence

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
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **Do not over-read this.** It is tempting — and this notebook's first draft did exactly that — to call
    the round-$k$ belief an *error bar on the computation*. It is not. It is the exact posterior of a
    **different problem**: the truncated $k$-hop computation tree. Nothing in it estimates the distance
    between $\mu^{(k)}$ and the answer $A^{-1}b$.

    The reason is visible in the update rule itself. The precision recursion
    $P_{ij} = -A_{ij}^2 / P_{i\setminus j}$ **contains no $b$**: the precisions form a closed system
    driven by the matrix alone. Change the right-hand side and every variance in this notebook is
    unchanged to the last bit, while the errors are completely different. So the belief cannot be
    tracking the error, and the two converge on schedules that have nothing to do with each other.

    That gap — an anytime *belief* that is not an anytime *error estimate* — is, to us, the most
    interesting open problem in this whole construction, and §5 comes back to it.
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(bump_forcing, gabp, grid_matrix, np, spla):
    m_grid = 24                       # 24 × 24 lattice, n = 576 unknowns
    screen_grid = 0.4
    A_grid = grid_matrix(m_grid, screening=screen_grid)
    b_grid = bump_forcing(m_grid)
    x_grid = spla.spsolve(A_grid.tocsc(), b_grid)
    grid_bp = gabp(A_grid, b_grid, iters=400, tol=1e-12, record=True)
    var_grid = np.diag(np.linalg.inv(A_grid.toarray()))     # reference marginals (dense, n = 576)
    return A_grid, b_grid, grid_bp, m_grid, var_grid, x_grid


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
    **What to look for.** At $k=1$ every node reports $b_i/A_{ii}$ — its own equation, nothing else — and a uniformly small standard deviation: maximal over-confidence. As rounds pass, the mean fills in from the sources outward, and the standard-deviation map inflates from the boundary inward, because nodes near the boundary genuinely *are* better determined (Dirichlet conditions pin them) while interior nodes must wait to learn how loosely they are held. Both fields stop changing once the information has travelled a correlation length.
    """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.6 Loops: exact means, over-confident variances

    On a graph with cycles the same information arrives at a node by several routes and gets double-counted. The remarkable fact (Weiss & Freeman 2001) is that this does **not** spoil the means: *if* GaBP converges, the marginal means are the exact solution $A^{-1}b$, cycles or no cycles. The variances are another matter — the computation tree that BP effectively solves keeps re-entering the same loop, and the walk-sum analysis of Malioutov, Johnson & Willsky (2006) shows BP counts only the self-return walks that revisit the root once. On an attractive model, where all those walks contribute with the same sign, the missing terms are positive, so BP **under-estimates** the variance: the solver is over-confident.

    That is the honest state of the art, and it is exactly the kind of statement the probabilistic-numerics community is equipped to improve on.
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
    ## 4.7 The punchline: Jacobi is GaBP with the uncertainty deleted

    Take the algorithm above and make two changes:

    1. clamp every precision message to zero, $P_{ij} := 0$;
    2. stop excluding the reverse message — let node $i$ use what $j$ told it when replying to $j$.

    What remains is $\mu_i = A_{ii}^{-1}\bigl(b_i - \sum_{k \neq i} A_{ki}\mu_k\bigr)$: **the Jacobi iteration** (Shental et al., Prop. 16). The classical stationary solver is the message-passing solver with the second moment thrown away and the cycle-avoidance thrown away.

    The classical method is not an alternative to the probabilistic one; it is the probabilistic one, marginalised down to a point estimate. Unlike Jacobi, GaBP carries precisions and excludes the reverse message. It is just *bookkeeping about information*, and it is what buys both the uncertainty estimate and the faster convergence.
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

    Identical to machine precision — and the two residuals show what the discarded second moment was worth.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.8 Scheduling: nobody has to wait

    * **Synchronous** — every node sends every round, using the previous round's messages. Like Jacobi.
    * **Asynchronous** — sweep the nodes and use each message the moment it exists. Like Gauss–Seidel.

    Neither needs an inner product, a norm, or any other quantity that couples all $n$ unknowns. Convergence is not destroyed by nodes running at different speeds, by stale messages, or by a node dropping out for a while. This is what makes the scheme viable on an unreliable, heterogeneous, or genuinely geographically distributed machine. Compare against the classical methods, remembering that each CG iteration hides two global barriers.
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
    mo.callout(
        mo.md(
            r"""
    **Reading the plot.** GaBP sits between the stationary methods and CG in iteration count — clearly better than Jacobi, comparable to or better than Gauss–Seidel — while being *strictly more local than either*: no global norm is ever formed, and the serial variant tolerates arbitrary update order. CG wins on iterations; whether it wins on wall-clock depends entirely on what a global reduction costs you. And on the tree, GaBP terminates *exactly* — in a bounded number of rounds, set by how far information must travel — which no stationary method does.
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.9 Does it scale?

    Per round, each node sends one two-scalar message per incident edge: the cost is $O(\mathrm{nnz})$ arithmetic and $O(\mathrm{nnz})$ communication, all of it nearest-neighbour, all of it parallel. So the only question that matters is **how the round count grows with $n$** — and that is where the probabilistic reading pays off in intuition.

    The screening parameter $c$ in $(c - \Delta)u = f$ sets the correlation length $\ell \sim 1/\sqrt{c}$ of the Gaussian field $\mathcal{N}(A^{-1}b, A^{-1})$. A node's marginal is determined by the nodes within a few $\ell$ of it; everything beyond is screened off. Information therefore has to travel a *fixed physical distance*, not across the whole domain. So the round count **saturates**: it stops growing with $n$, and the total work is $O(n)$ with perfect parallelism.

    At $c = 0$ the correlation length is the domain size, every node needs to hear from every other, and the round count grows like the diameter. This is not a defect of message passing; it is the same long-range coupling that makes unpreconditioned Jacobi and CG slow. It states what a preconditioner has to do: *shorten the correlation length*.
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
    ## 4.10 When it fails

    GaBP is not unconditionally convergent, and the sufficient conditions are the familiar ones:

    * $A$ strictly **diagonally dominant** $\Rightarrow$ convergence to the exact means (Weiss & Freeman 2001);
    * **walk-summability**, $\rho\bigl(|I - D^{-1}A|\bigr) < 1$ with $D = \operatorname{diag}(A)$, a strictly weaker condition (Malioutov et al. 2006);
    * a **tree**, in which case it converges exactly regardless of the spectral radius.

    In practice the basin is considerably larger than those conditions. But it does have an edge. Below, a lattice with random $\pm w$ couplings (a "frustrated" model, the sort where loops carry conflicting information) sweeps from harmless to divergent. Watch the diagnostics: the walk-summability bound is crossed long before anything goes wrong, and then convergence fails somewhere near the point where the Gaussian stops being a valid distribution at all.
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
    **Try it.** Push $w$ up from 0.05. The unit diagonal is beaten by the four couplings at $w = 0.25$ and walk-summability goes one step later at $w \approx 0.26$ — the two sufficient conditions fail together, and neither failure costs anything: the solver keeps converging, taking 45 rounds at $w = 0.26$ and 297 at $w = 0.29$. It breaks between $w = 0.29$ and $w = 0.30$, which is essentially where $A$ stops being positive definite ($\lambda_{\min} = +0.007$ at $w = 0.30$). Then turn on damping, $P \leftarrow (1-\alpha)P_{\text{new}} + \alpha P_{\text{old}}$: it buys smoothness in the borderline regime but does **not** rescue the indefinite case — and it should not, because there is no valid Gaussian left to infer. Sharp characterisations of the convergence basin, and principled fixes outside it, remain open (see Johnson et al. 2009, Ruozzi & Tatikonda 2013).
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 5. Where this goes

    We arrived at a linear solver that is local, asynchronous, communication-light, and reports a per-node uncertainty as a by-product — and whose classical counterpart (Jacobi) is literally itself with the second moment deleted. That combination is the argument of this tutorial: **probabilistic numerics at scale wants message passing, because message passing is what turns a global belief into a distributed one.**

    Possible research directions:

    * **Beyond symmetry.** GaBP as derived needs $A = A^\top$. Shental et al. (§VII) embed a rectangular $S$ into the symmetric system $\bigl(\begin{smallmatrix} I & S^\top \\ S & -\Psi \end{smallmatrix}\bigr)$, whose solution is the ridge/pseudo-inverse estimate $(S^\top S + \Psi)^{-1}S^\top y$ — with $2nk$ messages rather than $n^2$. Fanaskov (2022) instead modifies the messages themselves for non-symmetric $A$, relates the result to LU and block-LU factorisation, and uses GaBP as a **multigrid smoother**, where it is markedly more robust than incomplete-LU or Gauss–Seidel smoothing.
    * **Beyond linear systems.** Because an interior-point method is a sequence of linear systems (Newton steps on the Hessian), swapping each solve for GaBP gives a **distributed linear-programming solver** (Bickson et al. 2008). The same substitution works anywhere a Newton step is the inner loop.
    * **Better uncertainty.** Two distinct problems, and this tutorial solves neither. *First*, the means are exact on convergence but the variances are not: generalised BP / the cluster-variation method (the second algorithm in Fanaskov 2022), or the walk-sum corrections of Johnson et al., buy calibration by giving up locality. **What is the cheapest message-passing scheme with honest variances?** *Second*, and more fundamental (§4.5): the belief is a statement about the $k$-hop sub-problem, not about the error, because the precision recursion never sees $b$. An anytime *belief* is not an anytime *error estimate*, and nothing in the classical GaBP literature is trying to make it one. **What would a message that carried error information — rather than only information about $A$ — even look like?** Both are probabilistic-numerics questions, not linear-algebra ones; the second is the one we would most like an answer to.
    * **Applications where the graph is real.** Power-grid state estimation, sensor-network localisation, SLAM and bundle adjustment (Gaussian BP is the engine of several modern SLAM back-ends), CDMA multiuser detection — in each case the factor graph is not a metaphor for the sparsity pattern; it is the physical layout of the machine. The die of §1 is the limiting case, where the graph is *literally* the silicon; and that this is a good bargain in wall-clock, not just in rhetoric, has been measured. Ortiz et al. (2020) solved a real bundle-adjustment problem by GaBP on the 1216 cores of a single graph processor in under 40 ms, against 1450 ms for a sparse-Cholesky CPU library — the whole margin coming from an algorithm that never needs anything but nearest-neighbour exchange.

    <!-- ### Exercises

    1. **Elimination by hand.** Take a 5-node tree, run `gabp` for one sweep, and check that $P_{i\setminus j}$ equals the pivot $A_{ii} - \sum_{l} A_{li}^2/A_{ll}$ produced by Gaussian elimination from the leaves. (Shental et al., Prop. 14.)
    2. **Correlation length.** For the screened lattice, measure the round count as a function of $c$ and compare it against $\ell = 1/\sqrt{c}$ measured directly from the decay of $(A^{-1})_{ij}$ with $\|i - j\|$.
    3. **Preconditioning as re-modelling.** Apply a Jacobi and then an incomplete-Cholesky preconditioner and re-run GaBP on the preconditioned system. Explain the change in round count in terms of correlation length rather than condition number.
    4. **Anytime calibration.** For the loopy lattice, plot the actual error $|\mu_i^{(k)} - x_{\ast i}|$ against the belief standard deviation $\sqrt{1/P_i^{(k)}}$ at each round $k$. Is the $k$-round belief a usable stopping criterion? Where is it over-confident, and by how much?
    5. **Asynchrony.** Modify the serial schedule to update a random 30% of nodes per round, or to use messages one round stale. How much does the round count degrade? (This is the experiment that decides whether the method survives on real hardware.)
    6. **Non-symmetric.** Implement the augmented system of Shental et al. §VII and solve a rectangular least-squares problem by message passing. Compare against `scipy.sparse.linalg.lsqr`. -->

    ### References

    * Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867. Extended: [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
    * Bickson, D., Tock, Y., Shental, O., & Dolev, D. (2008). *Polynomial linear programming with Gaussian belief propagation*. Allerton, 895–901.
    * Fanaskov, V. (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear equations*. SIAM J. Sci. Comput., 44(2), A77–A102.
    * Weiss, Y., & Freeman, W. T. (2001). *Correctness of belief propagation in Gaussian graphical models of arbitrary topology*. Neural Computation, 13(10), 2173–2200.
    * Malioutov, D. M., Johnson, J. K., & Willsky, A. S. (2006). *Walk-sums and belief propagation in Gaussian graphical models*. JMLR, 7, 2031–2064.
    * Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press.
    * Cockayne, J., Oates, C. J., Ipsen, I. C. F., & Girolami, M. (2019). *A Bayesian conjugate gradient method*. Bayesian Analysis, 14(3), 937–1012.
    * Ortiz, J., Pupilli, M., Leutenegger, S., & Davison, A. J. (2020). *Bundle adjustment on a graph processor*. CVPR, 2413–2422. [arXiv:2003.03134](https://arxiv.org/abs/2003.03134).
    """)
    return


if __name__ == "__main__":
    app.run()
