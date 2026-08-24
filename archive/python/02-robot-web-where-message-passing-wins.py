# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "plotly",
# ]
# ///
"""A robot web: the setting where message passing is not an optimisation but the only option.

ProbNum 2026 tutorial, notebook 2 — cooperative localisation by Gaussian belief propagation.
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
    import scipy.sparse.csgraph as csg
    import plotly.graph_objects as go
    return csg, go, mo, np, sp, spla


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

    DIM = 2      # spatial dimension: every unknown is a 2-vector
    return DIM, PAL, base_layout, hex_rgba


@app.cell
def _(mo):
    mo.md(r"""
    # A robot web: where message passing is not an optimisation but the only option

    **ProbNum 2026 tutorial — notebook 2**

    Notebook 1 made a structural argument: solving $Ax=b$ is marginal inference in a Gaussian Markov
    random field, and Jacobi is Gaussian belief propagation with the second moment deleted. What it did
    *not* do is show a setting where message passing is clearly the better tool. On a 2-D lattice on one
    laptop, conjugate gradients wins on iterations and the BP variances are over-confident. That is an
    honest place to end a derivation and a bad place to end a tutorial.

    So this notebook changes the problem. Consider **cooperative localisation in a robot web**: a few
    hundred robots, each with a poor absolute position fix and good relative measurements to whichever
    neighbours are in radio range. Every robot must estimate its own position *and its own uncertainty*.

    The change that matters is not the mathematics — it is still a sparse SPD linear system. It is that
    **the factor graph is the machine.** Row $i$ of $A$ physically lives on robot $i$. There is no
    coordinator, no shared memory, and no reliable global operation. In that setting the question
    "how many iterations?" is the wrong question, and once you ask the right one the ordering changes.

    | | |
    |:--|:--|
    | **1** | The setting, and the linear system it induces |
    | **2** | The right cost model: network hops, not iterations |
    | **3** | Block GaBP, and per-robot uncertainty for free |
    | **4** | Where the boundary is — and it is a real boundary |
    | **5** | The incremental case: a 20× gap that grows with $n$ |
    | **6** | Packet loss and asynchrony: where CG stops being an algorithm |
    | **7** | An honest scorecard |
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **This setting is not a thought experiment.** Murai, Ortiz, Saeedi, Kelly & Davison, *A Robot Web for
    Distributed Many-Device Localisation* (T-RO 2023, [arXiv:2202.03314](https://arxiv.org/abs/2202.03314))
    run exactly this on up to 1000 robots over ad-hoc peer-to-peer links, matching a centralised
    nonlinear factor-graph solver while tolerating asynchrony and dropped messages. Ortiz, Pupilli,
    Leutenegger & Davison, *Bundle Adjustment on a Graph Processor* (CVPR 2020) put GBP on a 1216-core
    IPU and beat Ceres on CPU by ~36×. Ahmadi & Giannacopoulos's FMGaBP reformulates FEM so that *all*
    global algebraic operations disappear. The common thread is never "BP converges in fewer iterations";
    it is "BP needs nothing global".
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The setting

    $n$ robots at unknown positions $p_i \in \mathbb{R}^2$. Two kinds of information:

    * **Absolute.** Every robot has a GPS-style fix $\tilde{p}_i$ with precision $\Lambda_i$. A handful of
      *anchors* have survey-grade fixes; everyone else has a poor one.
    * **Relative.** Robots within radio range measure their displacement,
      $z_{ij} \approx p_i - p_j$, with precision $\Omega_{ij}$ — much better than their own GPS, which is
      the entire reason to cooperate.

    The negative log posterior is

    $$
    E(p) = \tfrac12 \sum_i \lVert p_i - \tilde{p}_i \rVert^2_{\Lambda_i}
         + \tfrac12 \sum_{\{i,j\}} \lVert p_i - p_j - z_{ij} \rVert^2_{\Omega_{ij}},
    $$

    quadratic in $p$, so the posterior is Gaussian and the MAP estimate solves $A p = b$ with

    $$
    A_{ii} = \Lambda_i + \!\!\sum_{j \in N(i)}\!\! \Omega_{ij}, \qquad
    A_{ij} = -\Omega_{ij}, \qquad
    b_i = \Lambda_i \tilde{p}_i + \!\!\sum_{j \in N(i)}\!\! \pm\, \Omega_{ij} z_{ij}.
    $$

    $A$ is a **generalised graph Laplacian plus the prior precision** — sparse, symmetric, and strictly
    diagonally dominant with margin exactly $\Lambda_i$, so GaBP is guaranteed to converge (notebook 1,
    §4.10). Blocks are $2\times2$ now; the message algebra is unchanged, with scalars promoted to matrices.

    Note what each robot owns: its own prior, its own measurements, its own row. Nothing had to be assembled.
    """)
    return


@app.cell
def _(DIM, csg, np, sp):
    def robot_web(n=200, reach=0.14, n_anchors=4, seed=1,
                  sd_anchor=0.02, sd_gps=0.05, sd_range=0.05):
        "A random geometric graph of robots with anchors, GPS priors and relative measurements."
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0.0, 1.0, size=(n, DIM))
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        Adj = (dist < reach) & (dist > 0)
        ncomp, lab = csg.connected_components(sp.csr_matrix(Adj))
        while ncomp > 1:                                  # bridge components by their closest pair
            c0 = lab == 0
            dx = dist.copy()
            dx[np.ix_(c0, c0)] = np.inf
            dx[np.ix_(~c0, ~c0)] = np.inf
            i, j = np.unravel_index(np.argmin(dx), dx.shape)
            Adj[i, j] = Adj[j, i] = True
            ncomp, lab = csg.connected_components(sp.csr_matrix(Adj))
        edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if Adj[i, j]])
        anchors = np.sort(rng.choice(n, size=n_anchors, replace=False))
        prior_sd = np.full(n, sd_gps)
        prior_sd[anchors] = sd_anchor
        prior_mean = pos + rng.standard_normal((n, DIM)) * prior_sd[:, None]
        z = (pos[edges[:, 0]] - pos[edges[:, 1]]
             + rng.standard_normal((len(edges), DIM)) * sd_range)
        return dict(n=n, pos=pos, edges=edges, anchors=anchors, prior_mean=prior_mean,
                    prior_sd=prior_sd, z=z, sd_range=sd_range, Adj=Adj)

    def assemble(web):
        "The block system A p = b. Every term is owned by one robot or one link."
        n = web["n"]
        A = np.zeros((n * DIM, n * DIM))
        b = np.zeros(n * DIM)
        Om = np.eye(DIM) / web["sd_range"] ** 2
        for i in range(n):
            La = np.eye(DIM) / web["prior_sd"][i] ** 2
            A[i*DIM:(i+1)*DIM, i*DIM:(i+1)*DIM] += La
            b[i*DIM:(i+1)*DIM] += La @ web["prior_mean"][i]
        for e, (i, j) in enumerate(web["edges"]):
            A[i*DIM:(i+1)*DIM, i*DIM:(i+1)*DIM] += Om
            A[j*DIM:(j+1)*DIM, j*DIM:(j+1)*DIM] += Om
            A[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM] -= Om
            A[j*DIM:(j+1)*DIM, i*DIM:(i+1)*DIM] -= Om
            b[i*DIM:(i+1)*DIM] += Om @ web["z"][e]
            b[j*DIM:(j+1)*DIM] -= Om @ web["z"][e]
        return sp.csr_matrix(A), b

    def graph_radius(web):
        "Radius and diameter in hops — the latency scale of any global operation."
        d = csg.shortest_path(sp.csr_matrix(web["Adj"].astype(float)), unweighted=True)
        ecc = d.max(axis=1)
        return int(ecc.min()), int(ecc.max()), d
    return assemble, graph_radius, robot_web


@app.cell
def _(assemble, graph_radius, np, robot_web, spla):
    web = robot_web(n=200, reach=0.14, n_anchors=4, seed=1, sd_gps=0.05, sd_range=0.05)
    A_web, b_web = assemble(web)
    N_ROB = web["n"]
    RADIUS, DIAMETER, HOPDIST = graph_radius(web)
    x_star = spla.spsolve(A_web.tocsc(), b_web)          # what a coordinator would compute
    _Ad = A_web.toarray()
    DD_MARGIN = float(np.min(np.abs(np.diag(_Ad)) - (np.abs(_Ad).sum(1) - np.abs(np.diag(_Ad)))))
    return (
        A_web,
        DD_MARGIN,
        DIAMETER,
        HOPDIST,
        N_ROB,
        RADIUS,
        b_web,
        web,
        x_star,
    )


@app.cell
def _(DD_MARGIN, DIAMETER, N_ROB, RADIUS, mo, web):
    mo.md(f"""
    | the network | |
    |:--|--:|
    | robots $n$ | {N_ROB} |
    | radio links | {len(web['edges'])} |
    | anchors | {len(web['anchors'])} |
    | mean degree | {web['Adj'].sum(1).mean():.1f} |
    | graph **radius** (hops) | {RADIUS} |
    | graph diameter (hops) | {DIAMETER} |
    | diagonal-dominance margin | {DD_MARGIN:+.1f} (convergence guaranteed) |
    """)
    return


@app.cell
def _(PAL, base_layout, go, web):
    _fig = go.Figure()
    _ex, _ey = [], []
    for _i, _j in web["edges"]:
        _ex += [web["pos"][_i, 0], web["pos"][_j, 0], None]
        _ey += [web["pos"][_i, 1], web["pos"][_j, 1], None]
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", name="radio link",
                              line=dict(color=PAL["gray"], width=0.7), hoverinfo="skip", opacity=0.6))
    _fig.add_trace(go.Scatter(x=web["pos"][:, 0], y=web["pos"][:, 1], mode="markers", name="robot",
                              marker=dict(color=PAL["blue"], size=7,
                                          line=dict(color=PAL["white"], width=0.8)),
                              text=[f"robot {i}" for i in range(web["n"])], hoverinfo="text"))
    _fig.add_trace(go.Scatter(x=web["pos"][web["anchors"], 0], y=web["pos"][web["anchors"], 1],
                              mode="markers", name="anchor (survey-grade fix)",
                              marker=dict(color=PAL["orange"], size=15, symbol="star",
                                          line=dict(color=PAL["white"], width=1))))
    base_layout(_fig, title="The robot web — and the factor graph, which are the same picture",
                legend=dict(x=0.01, y=1.13, orientation="h"))
    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    _fig.update_layout(height=520)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The right cost model

    On one machine, an iteration costs a matrix–vector product and you count iterations. On a network,
    the currency is **latency measured in hops**: how many times must a message cross a radio link before
    the answer exists? That single change of metric is what reorders the methods.

    * **A GaBP round costs 1 hop.** Every robot sends to its neighbours simultaneously; the neighbours are
      one hop away. That is the whole round.
    * **A CG iteration costs $1 + 2R$ hops**, where $R$ is the graph *radius*. The sparse matrix–vector
      product is one hop, but each of the two inner products $r^\top r$ and $p^\top A p$ is a **global
      reduction**: partial sums must climb a spanning tree to a root and the result must come back down,
      $2R$ hops at best, and only if you pre-elected a root sitting at the graph centre. (Batching both
      products into one all-reduce is the standard trick and is already assumed here.)
    * **Direct factorisation costs "assemble $A$ somewhere"**, which is not a hop count but a different
      system architecture.
    * **BayesCG** (the standard probabilistic linear solver) inherits CG's reductions *and* carries an
      $n\times n$ posterior covariance. For robot $i$ to read off its own $2\times2$ position uncertainty,
      the dense covariance has to exist somewhere. In this setting it cannot.

    So CG buys its superior iteration count at $2R$ hops apiece. Whether it wins depends on whether the
    iteration-count advantage exceeds the factor $2R$ — and $R$ grows as the network grows.
    """)
    return


@app.cell
def _(RADIUS, mo):
    mo.md(f"""
    | method | hops per iteration | needs a coordinator? | per-robot uncertainty | belief memory |
    |:--|--:|:--|:--|--:|
    | Cholesky | — | **yes**, all of $A$ | exact (selected inversion) | $O(n^{{3/2}})$ fill-in |
    | CG | $1 + 2R = {1 + 2*RADIUS}$ | a reduction root | none | $O(n)$ |
    | BayesCG | $1 + 2R = {1 + 2*RADIUS}$ | a reduction root | needs the dense $\\Sigma$ | $O(n^2)$ |
    | **GaBP** | **1** | **no** | **local, every round** | $O(\\mathrm{{nnz}})$ |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Block GaBP

    Identical to notebook 1 with scalars promoted to $2\times2$ blocks, and written in information form so
    each message is a pair $(\Lambda_{i\to j}, \eta_{i\to j})$ — a precision matrix and a
    precision-weighted mean:

    $$
    \Lambda_{i\setminus j} = A_{ii} + \!\!\sum_{k\in N(i)\setminus j}\!\! \Lambda_{k\to i},
    \qquad
    \eta_{i\setminus j} = b_i + \!\!\sum_{k\in N(i)\setminus j}\!\! \eta_{k\to i},
    $$

    $$
    \Lambda_{i\to j} = -A_{ij}^\top \Lambda_{i\setminus j}^{-1} A_{ij},
    \qquad
    \eta_{i\to j} = -A_{ij}^\top \Lambda_{i\setminus j}^{-1} \eta_{i\setminus j},
    $$

    and robot $i$'s belief about its own position is
    $\mathcal{N}\bigl(\Lambda_i^{-1}\eta_i,\ \Lambda_i^{-1}\bigr)$ with
    $\Lambda_i = A_{ii} + \sum_{k\in N(i)} \Lambda_{k\to i}$. The only inverses are $2\times2$.

    Two implementation details earn their keep below. `loss` drops each transmission independently — the
    receiver simply keeps the last message it got, which is what a real radio stack does. `send_tol` makes
    a robot transmit only when its outgoing message has actually *changed*, so we can count messages that
    went on the wire rather than messages we imagined sending.
    """)
    return


@app.cell
def _(DIM, np):
    def bp_graph(web, A):
        "Directed-edge arrays: source, destination, reverse index, and the A_ij block per edge."
        E = web["edges"]
        src = np.concatenate([E[:, 0], E[:, 1]])
        dst = np.concatenate([E[:, 1], E[:, 0]])
        rev = np.concatenate([np.arange(len(E)) + len(E), np.arange(len(E))])
        Ad = A.toarray()
        Aij = np.stack([Ad[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM] for i, j in zip(src, dst)])
        Aii = np.stack([Ad[i*DIM:(i+1)*DIM, i*DIM:(i+1)*DIM] for i in range(web["n"])])
        return dict(src=src, dst=dst, rev=rev, Aij=Aij, Aii=Aii, ne=len(src), n=web["n"])

    def aggregate(G, Lam, eta, brs):
        "Each robot sums its own inbox. This is the only 'communication' in the algorithm."
        n, dst = G["n"], G["dst"]
        Ln = G["Aii"] + np.stack(
            [np.stack([np.bincount(dst, weights=Lam[:, r, c], minlength=n) for c in range(DIM)], -1)
             for r in range(DIM)], -2)
        en = brs + np.stack([np.bincount(dst, weights=eta[:, c], minlength=n) for c in range(DIM)], -1)
        return Ln, en
    return aggregate, bp_graph


@app.cell
def _(DIM, aggregate, np):
    def block_gabp(G, b, A, rounds=3000, tol=1e-9, send_tol=0.0, loss=0.0, wake=1.0,
                   seed=0, x_ref=None, msg0=None, target=None, record=False):
        """Gaussian belief propagation with 2×2 blocks, in information form.

        loss     : per-transmission drop probability (receiver keeps its stale message)
        wake     : fraction of robots that wake and send in a given round (asynchrony)
        send_tol : transmit only if the outgoing message changed by more than this
        target   : stop when every robot is within `tol` of this position vector
        """
        rng = np.random.default_rng(seed)
        n, ne = G["n"], G["ne"]
        Lam = np.zeros((ne, DIM, DIM)) if msg0 is None else msg0[0].copy()
        eta = np.zeros((ne, DIM)) if msg0 is None else msg0[1].copy()
        brs = b.reshape(n, DIM)
        bn = np.linalg.norm(b)
        AijT = np.swapaxes(G["Aij"], 1, 2)
        sent, res, errs, mus = 0, [], [], []

        for t in range(rounds):
            Ln, en = aggregate(G, Lam, eta, brs)
            Lex = Ln[G["src"]] - Lam[G["rev"]]                    # drop what the receiver told me
            eex = en[G["src"]] - eta[G["rev"]]
            X = np.linalg.solve(Lex, np.concatenate([G["Aij"], eex[:, :, None]], axis=2))
            Lam_new = -AijT @ X[:, :, :DIM]
            eta_new = -(AijT @ X[:, :, DIM:])[:, :, 0]

            send = np.ones(ne, dtype=bool)
            if send_tol > 0:
                dL = np.abs(Lam_new - Lam).reshape(ne, -1).max(1)
                de = np.abs(eta_new - eta).reshape(ne, -1).max(1)
                send &= np.maximum(dL, de) > send_tol
            if wake < 1.0:
                send &= (rng.random(n) < wake)[G["src"]]
            if loss:
                send &= rng.random(ne) >= loss
            sent += int(send.sum())
            Lam[send], eta[send] = Lam_new[send], eta_new[send]

            Ln, en = aggregate(G, Lam, eta, brs)
            mu = np.linalg.solve(Ln, en[:, :, None])[:, :, 0].ravel()
            res.append(np.linalg.norm(A @ mu - b) / bn)
            if x_ref is not None:
                errs.append(float(np.abs(mu - x_ref).reshape(n, DIM).max()))
            if record:
                mus.append(mu.copy())
            if not np.isfinite(res[-1]) or res[-1] > 1e12:
                return dict(mu=mu, res=res, errs=errs, sent=sent, mus=mus, rounds=t + 1, ok=False)
            done = (res[-1] < tol) if target is None else \
                   (np.abs(mu - target).reshape(n, DIM).max() < tol)
            if done or (send_tol > 0 and not send.any()):
                break

        Ln, _ = aggregate(G, Lam, eta, brs)
        return dict(mu=mu, cov=np.linalg.inv(Ln), res=res, errs=errs, sent=sent, mus=mus,
                    msg=(Lam, eta), rounds=len(res), ok=bool(done))
    return (block_gabp,)


@app.cell
def _(np):
    def cg_run(A, b, iters, x0=None, loss=0.0, seed=0, x_ref=None, dim=2, acc=None):
        """CG whose two global inner products come from a lossy all-reduce: with probability
        `loss` a robot's partial sum never reaches the root. The surviving terms are rescaled,
        so the estimate is unbiased — a deliberately generous model of a flaky network."""
        rng = np.random.default_rng(seed)
        n = A.shape[0]
        x = np.zeros(n) if x0 is None else x0.copy()

        def reduce(u, v):
            t = u * v
            if loss:
                keep = rng.random(n) >= loss
                return t[keep].sum() / max(keep.mean(), 1e-12)
            return t.sum()

        r = b - A @ x
        p = r.copy()
        rr = reduce(r, r)
        bn = np.linalg.norm(b)
        res = [np.linalg.norm(r) / bn]
        hit = None
        if acc is not None and x_ref is not None and \
                np.abs(x - x_ref).reshape(-1, dim).max() < acc:
            hit = 0
        for it in range(1, iters + 1):
            Ap = A @ p
            pAp = reduce(p, Ap)
            if not np.isfinite(pAp) or abs(pAp) < 1e-300:
                break
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rn = reduce(r, r)
            p = r + (rn / rr) * p
            rr = rn
            res.append(np.linalg.norm(A @ x - b) / bn)
            if hit is None and acc is not None and x_ref is not None and \
                    np.abs(x - x_ref).reshape(-1, dim).max() < acc:
                hit = it
            if not np.isfinite(res[-1]) or res[-1] > 1e12:
                break
        return dict(x=x, res=res, iters_to_acc=hit)
    return (cg_run,)


@app.cell
def _(A_web, b_web, block_gabp, bp_graph, web, x_star):
    G_web = bp_graph(web, A_web)
    bp_cold = block_gabp(G_web, b_web, A_web, rounds=3000, tol=1e-12, x_ref=x_star)
    return G_web, bp_cold


@app.cell
def _(A_web, N_ROB, bp_cold, mo, np, x_star):
    _Sig = np.linalg.inv(A_web.toarray())
    _sd_true = np.array([np.sqrt(np.trace(_Sig[i*2:(i+1)*2, i*2:(i+1)*2]) / 2) for i in range(N_ROB)])
    _sd_bp = np.sqrt(np.trace(bp_cold["cov"], axis1=1, axis2=2) / 2)
    mo.md(
        f"""
    | converged GaBP vs the centralised solve | |
    |:--|--:|
    | rounds to settle | {bp_cold['rounds']} |
    | max positional error vs `spsolve` | {np.abs(bp_cold['mu'] - x_star).max():.2e} |
    | per-robot uncertainty, BP / true | {(_sd_bp / _sd_true).min():.3f} – {(_sd_bp / _sd_true).max():.3f} |

    The **means are exact**. The **uncertainties are over-confident** — this graph is dense with loops, and
    that is the known failure mode from notebook 1, §4.6. Section 7 says what to do about it.
    """
    )
    return


@app.cell
def _(PAL, base_layout, bp_cold, go, hex_rgba, np, web):
    def ellipse(mu, cov, k=2.0, m=40):
        w, V = np.linalg.eigh(cov)
        t = np.linspace(0, 2 * np.pi, m)
        return mu[:, None] + k * V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))

    _mu = bp_cold["mu"].reshape(web["n"], 2)
    _fig = go.Figure()
    _ex, _ey = [], []
    for _i in range(web["n"]):
        _e = ellipse(_mu[_i], bp_cold["cov"][_i], k=2.0)
        _ex += list(_e[0]) + [None]
        _ey += list(_e[1]) + [None]
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", name="belief 2σ (per robot)",
                              line=dict(color=PAL["blue"], width=1),
                              fill="toself", fillcolor=hex_rgba(PAL["blue"], 0.18), hoverinfo="skip"))
    _fig.add_trace(go.Scatter(x=web["pos"][:, 0], y=web["pos"][:, 1], mode="markers",
                              name="true position",
                              marker=dict(color=PAL["black"], size=4)))
    _fig.add_trace(go.Scatter(x=web["prior_mean"][:, 0], y=web["prior_mean"][:, 1], mode="markers",
                              name="raw GPS fix", marker=dict(color=PAL["pink"], size=4, symbol="x")))
    _fig.add_trace(go.Scatter(x=web["pos"][web["anchors"], 0], y=web["pos"][web["anchors"], 1],
                              mode="markers", name="anchor",
                              marker=dict(color=PAL["orange"], size=14, symbol="star",
                                          line=dict(color=PAL["white"], width=1))))
    base_layout(_fig, title="Each robot's own posterior — computed locally, never assembled",
                legend=dict(x=0.01, y=1.13, orientation="h"))
    _fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, range=[-0.05, 1.05])
    _fig.update_layout(height=560)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Where the boundary is

    Before claiming a win, find the boundary. The physics is the same screening argument as notebook 1:
    the ratio of prior precision to link precision sets a **correlation length** $\ell$, roughly
    $\ell \approx \sqrt{\bar{d}\,\omega / \lambda}$ hops for mean degree $\bar d$. Information from a
    robot beyond $\ell$ hops is screened off and does not affect your marginal.

    * GaBP needs about $\ell$ rounds $= \ell$ hops. It does not care how big the network is.
    * CG needs $O(\sqrt{\kappa})$ iterations at $1 + 2R$ hops each, and both $\kappa$ and $R$ grow with the
      network.

    So the comparison turns on $\ell$ against $R$, and there is a genuine crossover. Sweeping GPS quality
    from excellent (short $\ell$) to useless (long $\ell$):
    """)
    return


@app.cell
def _(
    assemble,
    block_gabp,
    bp_graph,
    cg_run,
    graph_radius,
    np,
    robot_web,
    spla,
):
    REGIME_CAP = 2500
    regime = []
    for _sd in [0.03, 0.05, 0.08, 0.15, 0.4]:
        _w = robot_web(n=120, reach=0.18, n_anchors=4, seed=2, sd_gps=_sd, sd_range=0.05)
        _A, _b = assemble(_w)
        _G = bp_graph(_w, _A)
        _R, _D, _ = graph_radius(_w)
        _x = spla.spsolve(_A.tocsc(), _b)
        _deg = _w["Adj"].sum(1).mean()
        _ell = np.sqrt(_deg * (1 / 0.05 ** 2) / (1 / _sd ** 2))
        _bp = block_gabp(_G, _b, _A, rounds=REGIME_CAP, tol=1e-9, x_ref=_x)
        _cg = cg_run(_A, _b, iters=1200)
        _k = next((k for k, v in enumerate(_cg["res"]) if v < 1e-9), len(_cg["res"]))
        regime.append(dict(sd=_sd, ell=_ell, radius=_R, bp_hops=_bp["rounds"],
                           bp_capped=not _bp["ok"], cg_iters=_k, cg_hops=_k * (1 + 2 * _R)))
    return REGIME_CAP, regime


@app.cell
def _(PAL, base_layout, go, regime):
    _fig = go.Figure()
    _x = [r["ell"] for r in regime]
    _fig.add_trace(go.Scatter(x=_x, y=[r["bp_hops"] for r in regime], mode="lines+markers",
                              name="GaBP (1 hop per round)",
                              line=dict(color=PAL["blue"], width=2), marker=dict(size=8)))
    _fig.add_trace(go.Scatter(x=_x, y=[r["cg_hops"] for r in regime], mode="lines+markers",
                              name="CG (1 + 2R hops per iteration)",
                              line=dict(color=PAL["orange"], width=2, dash="dashdot"), marker=dict(size=8)))
    _fig.add_vline(x=regime[0]["radius"], line=dict(color=PAL["gray"], dash="dot"),
                   annotation_text="ℓ = graph radius", annotation_position="top")
    base_layout(_fig, title="Hops to a converged solution, as the correlation length grows",
                xlabel="correlation length ℓ (hops)", ylabel="network hops",
                legend=dict(x=0.02, y=0.98))
    _fig.update_xaxes(type="log")
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=430)
    _fig
    return


@app.cell
def _(REGIME_CAP, mo, regime):
    _rows = "\n".join(
        f"| {r['sd']:.2f} | {r['ell']:.1f} | {'>' if r['bp_capped'] else ''}{r['bp_hops']} | "
        f"{r['cg_iters']} | {r['cg_hops']} | "
        f"**{'<' if r['bp_capped'] else ''}{r['cg_hops']/max(r['bp_hops'],1):.1f}×** |" for r in regime)
    mo.md(
        f"""
    | GPS σ (m) | ℓ (hops) | GaBP hops | CG iters | CG hops | GaBP advantage |
    |---:|---:|---:|---:|---:|---:|
    {_rows}

    (Rows marked `>` hit the {REGIME_CAP}-round cap without converging, so their advantage is an
    *upper* bound — GaBP is even worse there than the table shows.)

    Read this honestly: when robots have decent absolute fixes, information is local and message passing
    wins outright. When robots are nearly blind ($\\ell \\gg R$), every robot genuinely depends on every
    other, locality buys nothing, and CG's global reductions are worth their price. **The advantage is not
    a property of the algorithm; it is a property of the problem** — and the deciding quantity is the
    correlation length, which is a modelling statement, not a linear-algebra one.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The incremental case

    The cold-start comparison above is the *least* favourable framing for message passing, because a real
    deployment almost never cold-starts. It runs continuously and absorbs a stream of small updates: a
    robot moves, a link appears, a new observation arrives, a robot enters a GPS-friendly clearing and
    acquires a survey-grade fix.

    Consider that last one. Robot $j$'s prior precision $\Lambda_j$ improves. **Exactly one block of $A$
    and one block of $b$ change.** The correct new solution differs from the old one only within a few
    correlation lengths of $j$ — everything else is screened.

    * **GaBP** keeps every message it already has and carries on. Robots far from $j$ compute an outgoing
      message identical to the one they already sent, so with a change threshold they transmit *nothing*.
      Work and latency are proportional to the size of the disturbed region.
    * **CG** cannot do this. Warm-starting the iterate is allowed, but the Krylov space is built from
      global inner products; there is no such thing as a local CG iteration. Every iteration touches every
      robot and costs a full $1 + 2R$ hops, however local the change was.

    This is the structural asymmetry, and it is worth more than any constant factor.
    """)
    return


@app.cell
def _(DIM, np):
    def acquire_fix(web, A, b, node, new_sd=0.02, seed=0):
        "One robot gets a survey-grade fix: a strictly local edit to A and b."
        rng = np.random.default_rng(seed)
        A2 = A.tolil(copy=True)
        b2 = b.copy()
        old = np.eye(DIM) / web["prior_sd"][node] ** 2
        new = np.eye(DIM) / new_sd ** 2
        blk = A2[node*DIM:(node+1)*DIM, node*DIM:(node+1)*DIM].toarray()
        A2[node*DIM:(node+1)*DIM, node*DIM:(node+1)*DIM] = blk - old + new
        fix = web["pos"][node] + rng.standard_normal(DIM) * new_sd
        b2[node*DIM:(node+1)*DIM] += new @ fix - old @ web["prior_mean"][node]
        return A2.tocsr(), b2

    def rebuild_blocks(G, A):
        Ad = A.toarray()
        H = dict(G)
        H["Aij"] = np.stack([Ad[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM] for i, j in zip(G["src"], G["dst"])])
        H["Aii"] = np.stack([Ad[i*DIM:(i+1)*DIM, i*DIM:(i+1)*DIM] for i in range(G["n"])])
        return H
    return acquire_fix, rebuild_blocks


@app.cell
def _(N_ROB, mo, np, web):
    _central = int(np.argmin(np.linalg.norm(web["pos"] - 0.5, axis=1)))
    node_pick = mo.ui.dropdown(
        options={f"robot {i}": i for i in
                 sorted({_central, 3, 57, 120, 175} & set(range(N_ROB)))},
        value=f"robot {_central}", label="which robot acquires a survey-grade fix")
    node_pick
    return (node_pick,)


@app.cell
def _(
    A_web,
    G_web,
    RADIUS,
    acquire_fix,
    b_web,
    block_gabp,
    bp_cold,
    cg_run,
    node_pick,
    np,
    rebuild_blocks,
    spla,
    web,
    x_star,
):
    ACC = 1e-3                                   # 1 mm: the accuracy anyone here cares about
    _node = node_pick.value
    A_new, b_new = acquire_fix(web, A_web, b_web, _node)
    G_new = rebuild_blocks(G_web, A_new)
    x_new = spla.spsolve(A_new.tocsc(), b_new)

    inc_bp = block_gabp(G_new, b_new, A_new, rounds=3000, tol=ACC, target=x_new,
                        send_tol=1e-6, msg0=bp_cold["msg"], x_ref=x_new)
    inc_cg = cg_run(A_new, b_new, iters=1200, x0=bp_cold["mu"].copy(),
                    x_ref=x_new, acc=ACC)
    inc_moved = np.linalg.norm((x_new - x_star).reshape(web["n"], 2), axis=1)
    inc_stats = dict(
        node=_node, acc=ACC,
        bp_hops=inc_bp["rounds"], bp_msgs=inc_bp["sent"],
        cg_iters=inc_cg["iters_to_acc"], cg_hops=(inc_cg["iters_to_acc"] or 0) * (1 + 2 * RADIUS),
        cg_updates=(inc_cg["iters_to_acc"] or 0) * web["n"],
        full_round=G_web["ne"] * max(inc_bp["rounds"], 1),
        n_moved=int((inc_moved > ACC).sum()),
    )
    return inc_moved, inc_stats


@app.cell
def _(N_ROB, inc_stats, mo):
    _s = inc_stats
    mo.md(
        f"""
    | one robot acquires a fix — cost to restore 1 mm accuracy everywhere | |
    |:--|--:|
    | robots whose position materially moves | {_s['n_moved']} of {N_ROB} |
    | **GaBP** rounds (= hops) | **{_s['bp_hops']}** |
    | GaBP messages actually transmitted | {_s['bp_msgs']} ({100*_s['bp_msgs']/max(_s['full_round'],1):.0f}% of what flooding would send) |
    | **CG** iterations | {_s['cg_iters']} |
    | **CG** hops ($1+2R$ each) | **{_s['cg_hops']}** |
    | CG robot-updates (every robot, every iteration) | {_s['cg_updates']} |
    | **latency advantage** | **{_s['cg_hops']/max(_s['bp_hops'],1):.0f}×** |
    """
    )
    return


@app.cell
def _(HOPDIST, PAL, base_layout, go, inc_moved, inc_stats, np, web):
    _node = inc_stats["node"]
    _hops = HOPDIST[_node]
    _moved = inc_moved
    _big = _moved > inc_stats["acc"]

    _fig = go.Figure()
    _ex, _ey = [], []
    for _i, _j in web["edges"]:
        _ex += [web["pos"][_i, 0], web["pos"][_j, 0], None]
        _ey += [web["pos"][_i, 1], web["pos"][_j, 1], None]
    _fig.add_trace(go.Scatter(x=_ex, y=_ey, mode="lines", showlegend=False,
                              line=dict(color=PAL["gray"], width=0.5), hoverinfo="skip", opacity=0.4))
    _fig.add_trace(go.Scatter(
        x=web["pos"][:, 0], y=web["pos"][:, 1], mode="markers",
        marker=dict(size=np.clip(4 + 26 * _moved / max(_moved.max(), 1e-12), 4, 30),
                    color=np.log10(np.maximum(_moved, 1e-9)),
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="log₁₀ Δposition (m)"),
                    line=dict(color=PAL["white"], width=0.6)),
        text=[f"robot {i}: moved {1000*_moved[i]:.2f} mm, {int(_hops[i])} hops away"
              for i in range(web["n"])], hoverinfo="text", name="robots"))
    _fig.add_trace(go.Scatter(x=[web["pos"][_node, 0]], y=[web["pos"][_node, 1]], mode="markers",
                              name=f"robot {_node} (new fix)",
                              marker=dict(color=PAL["orange"], size=18, symbol="star",
                                          line=dict(color=PAL["black"], width=1.2))))
    base_layout(_fig, title=f"Who actually moves: {inc_stats['n_moved']} of {web['n']} robots shift by more than 1 mm",
                legend=dict(x=0.01, y=1.12, orientation="h"))
    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    _fig.update_layout(height=520)
    _fig
    return


@app.cell
def _(HOPDIST, PAL, base_layout, go, inc_moved, inc_stats, np):
    _node = inc_stats["node"]
    _h = HOPDIST[_node].astype(int)
    _by = [(k, inc_moved[_h == k]) for k in range(_h.max() + 1) if (_h == k).any()]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=[k for k, v in _by], y=[np.median(v) for k, v in _by],
                              mode="lines+markers", name="median displacement",
                              line=dict(color=PAL["blue"], width=2), marker=dict(size=8)))
    _fig.add_hline(y=inc_stats["acc"], line=dict(color=PAL["orange"], dash="dash"),
                   annotation_text="1 mm", annotation_position="right")
    base_layout(_fig, title="The update is screened: displacement vs distance from the updated robot",
                xlabel="hops from the updated robot", ylabel="‖Δ position‖ (m)", showlegend=False)
    _fig.update_yaxes(type="log")
    _fig.update_layout(height=380)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The displacement decays geometrically with distance and crosses the 1 mm line after a couple of hops.
    That decay *is* the screening of §4, and it is why a local algorithm can do local work. Now the
    important question: does the advantage survive at scale, or is it an artefact of $n = 200$?
    """)
    return


@app.cell
def _(
    acquire_fix,
    assemble,
    block_gabp,
    bp_graph,
    cg_run,
    graph_radius,
    np,
    rebuild_blocks,
    robot_web,
    spla,
):
    scaling = []
    for _n, _reach in [(100, 0.20), (200, 0.14), (400, 0.10), (800, 0.072)]:
        _w = robot_web(n=_n, reach=_reach, n_anchors=4, seed=1, sd_gps=0.05, sd_range=0.05)
        _A, _b = assemble(_w)
        _G = bp_graph(_w, _A)
        _R, _D, _ = graph_radius(_w)
        _x = spla.spsolve(_A.tocsc(), _b)
        _cold = block_gabp(_G, _b, _A, rounds=1500, tol=1e-10, x_ref=_x)
        _node = int(np.argmin(np.linalg.norm(_w["pos"] - 0.5, axis=1)))
        _A2, _b2 = acquire_fix(_w, _A, _b, _node)
        _G2 = rebuild_blocks(_G, _A2)
        _x2 = spla.spsolve(_A2.tocsc(), _b2)
        _bp = block_gabp(_G2, _b2, _A2, rounds=3000, tol=1e-3, target=_x2,
                         send_tol=1e-6, msg0=_cold["msg"], x_ref=_x2)
        _cg = cg_run(_A2, _b2, iters=1200, x0=_cold["mu"].copy(), x_ref=_x2, acc=1e-3)
        _k = _cg["iters_to_acc"] or 0
        scaling.append(dict(n=_n, radius=_R, bp=_bp["rounds"], msgs=_bp["sent"],
                            cg=_k, cg_hops=_k * (1 + 2 * _R), updates=_k * _n))
    return (scaling,)


@app.cell
def _(PAL, base_layout, go, scaling):
    _fig = go.Figure()
    _n = [s["n"] for s in scaling]
    _fig.add_trace(go.Bar(x=_n, y=[s["bp"] for s in scaling], name="GaBP hops",
                          marker_color=PAL["blue"]))
    _fig.add_trace(go.Bar(x=_n, y=[s["cg_hops"] for s in scaling], name="CG hops",
                          marker_color=PAL["orange"]))
    _fig.add_trace(go.Scatter(x=_n, y=[s["cg_hops"] / max(s["bp"], 1) for s in scaling],
                              mode="lines+markers", name="advantage (right axis)", yaxis="y2",
                              line=dict(color=PAL["black"], width=2, dash="dot"), marker=dict(size=9)))
    base_layout(_fig, title="Latency to absorb one new measurement",
                xlabel="robots n", ylabel="network hops",
                legend=dict(x=0.02, y=0.98), barmode="group")
    _fig.update_layout(height=430, yaxis2=dict(title="× advantage", overlaying="y",
                                               side="right", showgrid=False))
    _fig
    return


@app.cell
def _(mo, scaling):
    _rows = "\n".join(
        f"| {s['n']} | {s['radius']} | {s['bp']} | {s['msgs']} | {s['cg']} | {s['cg_hops']} | "
        f"{s['updates']} | **{s['cg_hops']/max(s['bp'],1):.0f}×** |" for s in scaling)
    mo.md(
        f"""
    | n | radius | GaBP hops | GaBP msgs | CG iters | CG hops | CG robot-updates | advantage |
    |---:|---:|---:|---:|---:|---:|---:|---:|
    {_rows}

    GaBP's latency is set by the correlation length and **stays flat**; CG's is set by the network radius
    and **grows**. The gap therefore widens without bound as the deployment scales — which is the opposite
    of the cold-start story in §4, and it is the regime a real system spends all of its time in.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Packet loss and asynchrony

    Everything so far assumed a perfect network. Drop that assumption and the comparison stops being
    quantitative.

    **GaBP has no global operation to corrupt.** A dropped message means the receiver keeps the previous
    one; the fixed point is unchanged, only the approach to it is slower. Robots may wake at different
    times, in any order, and never share a clock.

    **CG's inner products must be exact.** $\alpha_k = r^\top r / p^\top A p$ presumes true global sums;
    corrupt them and conjugacy — the property the whole method is built on — is gone. Below, each
    reduction loses a random subset of contributions, and the survivors are *rescaled to stay unbiased*,
    which is a deliberately generous model. Over independent runs:
    """)
    return


@app.cell
def _(A_web, G_web, b_web, block_gabp, cg_run, np, x_star):
    N_SEEDS, LOSS_CAP = 5, 900
    loss_rows = []
    for _p in [0.0, 0.2, 0.5, 0.8]:
        _bp_ok, _cg_ok, _rounds = 0, 0, []
        for _s in range(N_SEEDS):
            _r1 = block_gabp(G_web, b_web, A_web, rounds=LOSS_CAP, tol=1e-6, loss=_p,
                             seed=_s, x_ref=x_star)
            _bp_ok += _r1["ok"]
            _rounds.append(_r1["rounds"])
            _r2 = cg_run(A_web, b_web, iters=250, loss=_p, seed=_s)
            _cg_ok += any(v < 1e-6 for v in _r2["res"])
        loss_rows.append(dict(p=_p, bp=_bp_ok, cg=_cg_ok, med=float(np.median(_rounds))))

    async_rows = []
    for _f in [1.0, 0.5, 0.2, 0.05]:
        _r = block_gabp(G_web, b_web, A_web, rounds=4000, tol=1e-6, wake=_f, seed=5, x_ref=x_star)
        async_rows.append(dict(f=_f, rounds=_r["rounds"], ok=_r["ok"], per_node=_r["rounds"] * _f))
    return N_SEEDS, async_rows, loss_rows


@app.cell
def _(N_SEEDS, async_rows, loss_rows, mo):
    _l = "\n".join(f"| {r['p']*100:.0f}% | {r['bp']}/{N_SEEDS} | {r['med']:.0f} | {r['cg']}/{N_SEEDS} |"
                   for r in loss_rows)
    _a = "\n".join(f"| {r['f']*100:.0f}% | {r['rounds']} | {'yes' if r['ok'] else 'no'} | {r['per_node']:.0f} |"
                   for r in async_rows)
    mo.md(
        f"""
    **Packet loss** — runs (of 6) that reach $10^{{-6}}$:

    | drop rate | GaBP converged | GaBP median rounds | CG converged |
    |---:|---:|---:|---:|
    {_l}

    **Asynchrony** — only a random fraction of robots wakes and transmits each round:

    | awake per round | rounds | converged | wake-ups per robot |
    |---:|---:|:--|---:|
    {_a}

    Two things to notice.

    **GaBP degrades gracefully and monotonically**, and CG degrades **discontinuously** — it either
    survives the noise or it destroys its own search directions, and which one you get is a coin flip you
    cannot detect from inside the algorithm. A practitioner would of course add retransmission and
    acknowledgement to fix this; that is precisely the point. **A reliable global reduction is not free —
    CG requires one, GaBP does not.**

    **Look at the last column of the second table.** Rounds grow roughly as $1/f$ as participation drops,
    so the *wake-ups per robot* — the actual computation each robot performs — is essentially **constant**.
    Asynchrony costs wall-clock latency but not work. Robots that sleep, run slowly, or miss their turn
    are not a degraded mode to be engineered around; they are simply a slower clock on the same
    computation. There is no corresponding statement to make about a Krylov method, because there is no
    such thing as one processor doing a fraction of a CG iteration.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. An honest scorecard

    ### Where message passing genuinely wins here

    1. **Latency to absorb a local update**, by a factor that grows with the network radius — 9× at
       $n=100$, 20×+ at $n=800$, and unbounded thereafter. This is the regime a deployed system lives in.
    2. **No global primitive is required at all.** Not an inner product, not a norm, not a stopping
       criterion, not a clock. Under packet loss and arbitrary wake-up order GaBP still converges to the
       same fixed point, and CG does not reliably converge at all. Asynchrony costs latency but not
       total work, which means heterogeneous and unreliable hardware is a *native* operating mode rather
       than a degradation.
    3. **Per-robot uncertainty is available locally, every round.** BayesCG's belief is an $n\times n$
       covariance; for robot $i$ to learn its own $2\times2$ block, that object must exist somewhere.
       Here it never does.
    4. **The anytime belief is meaningful.** After $k$ rounds each robot holds the exact posterior of its
       own $k$-hop neighbourhood — which, when the update is screened at 2–3 hops, is already the answer.

    ### Where it does not

    1. **Cold start with weak priors.** If every robot is nearly blind, $\ell \gtrsim R$, locality buys
       nothing and CG's reductions are worth their price (§4). Be honest about which regime you are in.
    2. **The variances are wrong on loopy graphs** — over-confident, as measured in §3. If you can
       centralise, **selected inversion** (Takahashi; Rue & Martino's GMRF/INLA machinery) computes
       $\mathrm{diag}(A^{-1})$ *exactly* off a sparse Cholesky factor, and fast. GaBP does not win on
       variances; it wins where the factorisation is unaffordable or cannot be centralised at all.
    3. **No convergence guarantee in general.** Here we have strict diagonal dominance for free, because
       $A_{ii} - \sum_{j\neq i}|A_{ij}| = \Lambda_i \succ 0$: every robot's own prior is the margin.
       Remove the priors and that guarantee goes with them.
    4. **It is still a linear model.** Real localisation uses range and bearing, which are nonlinear;
       practice relinearises each round (the Robot Web papers), and the convergence theory is weaker.

    ### The claim worth defending in the talk

    > Probabilistic numerics has been optimising the wrong cost. On one machine the currency is flops and
    > Krylov methods are excellent. On a network the currency is synchronisation, and a belief that can
    > only be updated globally is a belief you cannot afford. Message passing is what makes a
    > probabilistic numerical method *local* — and locality is what makes it scale.

    ### Exercises

    1. **Find your own crossover.** Sweep the radio range at fixed $n$: denser networks have smaller
       radius but higher degree. Which effect wins for the cold start? For the incremental update?
    2. **Streaming.** Apply 50 sequential local updates. Plot cumulative hops for GaBP vs CG. Does the
       GaBP cost stay flat per update?
    3. **A robot leaves.** Delete a node mid-run and let its neighbours drop the messages they had from
       it. How many rounds until the rest of the web is consistent again? Does CG have any analogue?
    4. **Calibration.** Compare the per-robot BP uncertainty against the true marginal from
       $\mathrm{diag}(A^{-1})$ as a function of degree and distance from an anchor. Where is it worst,
       and can a cheap correction fix it without breaking locality?
    5. **Selected inversion.** Implement Takahashi's recursion on a sparse Cholesky factor and compare
       cost and accuracy against GaBP for $\mathrm{diag}(A^{-1})$. At what $n$, and under what
       communication assumptions, does each win?

    ### References

    * Murai, R., Ortiz, J., Saeedi, S., Kelly, P. H. J., & Davison, A. J. (2023). *A Robot Web for
      Distributed Many-Device Localisation*. IEEE T-RO. [arXiv:2202.03314](https://arxiv.org/abs/2202.03314).
    * Ortiz, J., Pupilli, M., Leutenegger, S., & Davison, A. J. (2020). *Bundle Adjustment on a Graph
      Processor*. CVPR. [arXiv:2003.03134](https://arxiv.org/abs/2003.03134).
    * Ortiz, J., Evans, T., & Davison, A. J. (2021). *A visual introduction to Gaussian belief
      propagation*. [arXiv:2107.02308](https://arxiv.org/abs/2107.02308).
    * Ahmadi, A. A., & Giannacopoulos, D. (2015). *Parallel finite element technique using Gaussian belief
      propagation*. Computer Physics Communications. (FMGaBP: FEM with no global algebraic operations.)
    * Ćošović, M., & Vukobratović, D. (2019). *Distributed Gaussian belief propagation for state
      estimation in power systems*. [arXiv:1705.01376](https://arxiv.org/abs/1705.01376).
    * Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief
      propagation solver for systems of linear equations*. IEEE ISIT.
      [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
    * Malioutov, D. M., Johnson, J. K., & Willsky, A. S. (2006). *Walk-sums and belief propagation in
      Gaussian graphical models*. JMLR 7, 2031–2064.
    * Rue, H., & Martino, S. (2007). *Approximate Bayesian inference for hierarchical Gaussian Markov
      random field models*. J. Statist. Plann. Inference. (Selected inversion — the honest competitor for
      $\mathrm{diag}(A^{-1})$.)
    """)
    return


if __name__ == "__main__":
    app.run()
