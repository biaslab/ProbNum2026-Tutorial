# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""CG against BayesCG: what the uncertainty costs, and what it is worth.

ProbNum 2026 tutorial, §3. Companion to `bayescg-animation.py`, which explains the
update; this one measures it. Two panels, both on the real die problem rather than
on the two-dimensional plane the animations draw.

**Accuracy.** With $\\Sigma_0 = A^{-1}$ and CG's search directions, BayesCG's posterior
mean *is* the CG iterate — not approximately, identically, and the left panel is
that claim measured over every iteration. So there is no accuracy to trade: the
point estimate is the same object arrived at by a different argument. Two things
do differ, and both are on the panel:

* Change the prior and you lose it. With $\\Sigma_0 = I$ and the same directions the
  mean is a different projection of the same information, and a worse one.
* BayesCG's own error estimate, $\\mathbb{E}\\|x-x_m\\|_A^2 = \\mathrm{tr}(A\\Sigma_m) = n-m$,
  decays linearly in the iteration count no matter what the error does. Plotted
  beside the actual error it is not an error bar; it is a countdown of directions.

**Runtime.** The mean is free and the covariance is the whole bill. With
$\\Sigma_0 = A^{-1}$ every $A^{-1}$ cancels out of the mean recursion — $\\Sigma_0 A S = S$
and $\\Lambda = S^\\top\\! A S$ is diagonal because the directions are conjugate — so
BayesCG's mean costs CG plus the storage of the directions, which is what the
middle curve shows. The covariance does not simplify:

    Sigma_m = A^{-1} - S (S^T A S)^{-1} S^T ,

whose rank-$m$ correction is cheap and whose first term is the inverse of $A$. The
prior that buys the exact mean is exactly the object nobody can afford. Worse, a
variance for one unknown needs one diagonal entry of $A^{-1}$, so there is no
cheap partial answer either — though `numpy`'s dense inverse is an upper bound on
that cost, not the last word: selected inversion off a sparse Cholesky (Takahashi;
Rue & Martino) gets $\\mathrm{diag}(A^{-1})$ far faster, and if the problem can be
centralised at all, that is how to get it.

Timings are best-of-`--repeats` wall clock on whatever machine runs the script, at
a common iteration count taken from CG's own convergence, so the three curves
differ only in what they compute and not in how far they got.

    uv run python/cg-vs-bayescg-figure.py                 # -> figures/cg-vs-bayescg.png
    uv run python/cg-vs-bayescg-figure.py --sizes 8,16,24 # a quicker pass

Without uv:  pip install numpy matplotlib  &&  python python/cg-vs-bayescg-figure.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = dict(blue="#2a78d6", red="#C81919", navy="#101073", gray="#898781",
           orange="#eb6834", green="#008300", violet="#4a3aa7")
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight"})


# ----------------------------------------------------------------- the die problem
def make_apply(m: int, c: float):
    """A = (c - Delta) as a matrix-free five-point stencil: O(nnz) per call."""
    def apply_A(v):
        u = v.reshape(m, m)
        out = (4.0 + c) * u
        out[1:, :] -= u[:-1, :]
        out[:-1, :] -= u[1:, :]
        out[:, 1:] -= u[:, :-1]
        out[:, :-1] -= u[:, 1:]
        return out.ravel()
    return apply_A


def dense_matrix(m: int, c: float) -> np.ndarray:
    """The same operator as an n x n array, for the parts that need one."""
    n = m * m
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            A[k, k] = 4.0 + c
            if j + 1 < m:
                A[k, k + 1] = A[k + 1, k] = -1.0
            if i + 1 < m:
                A[k, k + m] = A[k + m, k] = -1.0
    return A


def power_map(m: int, width: float = 0.18) -> np.ndarray:
    """Dissipated power b: two Gaussian workloads on the die."""
    g = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(g, g, indexing="ij")
    b = np.zeros((m, m))
    for cx, cy, amp in ((0.30, 0.32, 1.0), (0.70, 0.66, 0.8)):
        b += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
    return b


# ----------------------------------------------------------------- the two methods
def cg(apply_A, b, iters: int, tol: float = 0.0, keep_dirs: bool = False):
    """Conjugate gradients, matrix-free. keep_dirs is the whole of BayesCG's mean."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(r @ r)
    bn = float(np.linalg.norm(b))
    dirs, xs = [], [x.copy()]
    k = 0
    for k in range(1, iters + 1):
        Ap = apply_A(p)
        alpha = rr / float(p @ Ap)
        if keep_dirs:
            dirs.append(p.copy())
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(r @ r)
        xs.append(x.copy())
        if np.sqrt(rr_new) <= tol * bn:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return dict(x=x, iters=k, dirs=dirs, xs=xs)


def bayescg_means(A, b, S, prior: str):
    """Posterior means after 1..m observations, for Sigma_0 = A^{-1} or I.

    Sigma_0 = A^{-1}:  x_m = S (S^T A S)^{-1} S^T r_0  -- every A^{-1} has cancelled.
    Sigma_0 = I:       x_m = A S (S^T A^2 S)^{-1} S^T r_0, by least squares because
    S^T A^2 S goes ill-conditioned as m grows.

    S^T A S is diagonal in exact arithmetic, the directions being A-conjugate, but
    dividing by that diagonal is not the way to use the fact: its entries span some
    25 orders of magnitude by iteration 45 (p^T A p collapses as CG converges), so
    off-diagonals that are 1e-16 of the largest entry are enormous next to the
    smallest. The shortcut costs seven digits of agreement with CG; the solve costs
    nothing here and keeps it at 1e-16.
    """
    AS = A @ S
    means = []
    for m in range(1, S.shape[1] + 1):
        Sm, ASm = S[:, :m], AS[:, :m]
        rhs = Sm.T @ b
        if prior == "ainv":
            means.append(Sm @ np.linalg.solve(Sm.T @ ASm, rhs))
        else:
            means.append(ASm @ np.linalg.lstsq(ASm.T @ ASm, rhs, rcond=None)[0])
    return means


def a_norm(A, e):
    return float(np.sqrt(max(e @ (A @ e), 0.0)))


# ----------------------------------------------------------------- the measurements
def accuracy(args):
    """One problem, every iteration: CG, both BayesCG means, and BayesCG's estimate."""
    m, c = args.acc_size, args.screening
    n = m * m
    A = dense_matrix(m, c)
    b = power_map(m).ravel()
    x_star = np.linalg.solve(A, b)
    ref = a_norm(A, x_star)

    run = cg(make_apply(m, c), b, iters=args.acc_iters, tol=1e-13, keep_dirs=True)
    S = np.column_stack(run["dirs"])
    K = S.shape[1]

    cg_err = [a_norm(A, x - x_star) / ref for x in run["xs"][1:K + 1]]
    out = {"cg": np.array(cg_err)}
    for prior in ("ainv", "identity"):
        mus = bayescg_means(A, b, S, prior)
        out[prior] = np.array([a_norm(A, mu - x_star) / ref for mu in mus])
        if prior == "ainv":
            out["gap"] = np.array([np.linalg.norm(mu - x) / max(np.linalg.norm(x), 1e-300)
                                   for mu, x in zip(mus, run["xs"][1:K + 1])])
    # tr(A Sigma_m) = n - m, normalised the same way as the error curves:
    # sqrt(E||x-x_m||_A^2 / E||x-x_0||_A^2) = sqrt((n-m)/n)
    ms = np.arange(1, K + 1)
    out["claim"] = np.sqrt((n - ms) / n)
    out.update(n=n, m=m, K=K, ms=ms)
    return out


def timings(args):
    """Wall clock against problem size, at a common iteration count per size."""
    rows = []
    for m in args.sizes:
        n, c = m * m, args.screening
        apply_A = make_apply(m, c)
        b = power_map(m).ravel()
        A = dense_matrix(m, c)

        k = cg(apply_A, b, iters=20 * m, tol=args.tol)["iters"]

        def best(fn):
            """Best-of-repeats, timeit style: warm up, then batch until it is
            measurable. Without the batching a 0.4 ms CG is dominated by clock noise
            and BLAS scheduling, which showed up as Sigma_m timing faster than the
            inv(A) inside it."""
            fn()
            t0 = time.perf_counter()
            fn()
            inner = max(1, int(args.min_time / max(time.perf_counter() - t0, 1e-9)))
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                for _ in range(inner):
                    fn()
                ts.append((time.perf_counter() - t0) / inner)
            return min(ts)

        t_cg = best(lambda: cg(apply_A, b, iters=k))
        run = cg(apply_A, b, iters=k, keep_dirs=True)
        S = np.column_stack(run["dirs"])
        t_mean = best(lambda: _mean_only(apply_A, b, k))
        t_inv = best(lambda: np.linalg.inv(A))
        t_cov = best(lambda: _cov(A, S))

        rows.append(dict(m=m, n=n, k=k, cg=t_cg, mean=t_mean, cov=t_cov, inv=t_inv,
                         bytes_cov=8 * n * n, bytes_A=8 * int((A != 0).sum()),
                         bytes_S=8 * n * k))
        print(f"  n = {n:6d}  k = {k:3d}   CG {t_cg * 1e3:9.2f} ms   "
              f"BayesCG mean {t_mean * 1e3:9.2f} ms   "
              f"Sigma_m {t_cov * 1e3:10.2f} ms   (of which inv(A) "
              f"{t_inv * 1e3:9.2f} ms)")
    return rows


def _mean_only(apply_A, b, k):
    """BayesCG's posterior mean: CG's recursion, keeping the directions."""
    return cg(apply_A, b, iters=k, keep_dirs=True)


def _cov(A, S):
    """Sigma_m = A^{-1} - S (S^T A S)^{-1} S^T, formed explicitly.

    The rank-m correction is O(n m^2); inverting A is O(n^3) and is the whole bill.
    """
    Ainv = np.linalg.inv(A)
    return Ainv - S @ np.linalg.solve(S.T @ (A @ S), S.T)


# ----------------------------------------------------------------- the figure
def draw(acc, rows, args):
    fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=(10.6, 4.4),
                                     gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    fig.patch.set_facecolor("white")

    # ---------------- accuracy
    ms = acc["ms"]
    ax_a.semilogy(ms, acc["cg"], color=PAL["blue"], lw=3.0, alpha=0.85,
                  label="CG")
    ax_a.semilogy(ms, acc["ainv"], color=PAL["orange"], lw=1.6, ls=(0, (4, 3)),
                  label=r"BayesCG mean, $\Sigma_0 = A^{-1}$")
    ax_a.semilogy(ms, acc["identity"], color=PAL["green"], lw=1.6, ls="-.",
                  label=r"BayesCG mean, $\Sigma_0 = I$")
    ax_a.semilogy(ms, acc["claim"], color=PAL["red"], lw=1.6, ls=":",
                  label=r"BayesCG's own estimate, $\sqrt{(n-m)/n}$")
    ax_a.set_title(f"accuracy on the {acc['m']}×{acc['m']} die "
                   f"($n = {acc['n']}$)", fontsize=11.5, color="#222")
    ax_a.set_xlabel("iteration $m$", fontsize=10)
    ax_a.set_ylabel(r"$\|x_m - x^\ast\|_A\,/\,\|x^\ast\|_A$", fontsize=10)
    ax_a.set_xlim(1, acc["K"])
    ax_a.set_ylim(top=30.0)                 # headroom above the estimate curve
    ax_a.grid(alpha=0.25)
    ax_a.legend(fontsize=8.5, loc="lower left", framealpha=0.92, borderpad=0.35)
    ax_a.text(0.98, 0.66, f"CG and the $A^{{-1}}$ prior agree to "
              f"{acc['gap'].max():.0e}\nover all {acc['K']} iterations",
              transform=ax_a.transAxes, fontsize=9.0, color=PAL["navy"],
              ha="right", va="top")
    # the identity-prior curve stalls where its projection stops being solvable,
    # which is a numerical fact about that formula, not a property of the prior
    j = int(0.80 * acc["K"])
    ax_a.text(j, acc["identity"][j - 1] / 5.0,
              r"stalls: $S^\top\! A^2 S$ singular", color=PAL["green"],
              fontsize=8.5, ha="center", va="top")

    # ---------------- runtime
    n = np.array([r["n"] for r in rows], dtype=float)
    t_cg = np.array([r["cg"] for r in rows])
    t_mean = np.array([r["mean"] for r in rows])
    t_cov = np.array([r["cov"] for r in rows])
    # fitted slopes rather than idealised guide lines: at these sizes neither curve
    # is in its asymptotic regime -- CG is dominated by per-call overhead and LAPACK's
    # inverse is still memory-bound -- and drawing n^1 and n^3 through the data would
    # imply agreement that is not there
    fit = lambda t: np.polyfit(np.log(n[-4:]), np.log(t[-4:]), 1)[0]
    ax_t.loglog(n, t_cg * 1e3, "-o", color=PAL["blue"], lw=2.2, ms=5,
                label=f"CG (matrix-free),  $\\propto n^{{{fit(t_cg):.1f}}}$")
    ax_t.loglog(n, t_mean * 1e3, "-s", color=PAL["orange"], lw=1.8, ms=4.5,
                label=f"BayesCG, posterior mean,  $\\propto n^{{{fit(t_mean):.1f}}}$")
    ax_t.loglog(n, t_cov * 1e3, "-^", color=PAL["red"], lw=2.2, ms=5,
                label=f"BayesCG, full $\\Sigma_m$,  $\\propto n^{{{fit(t_cov):.1f}}}$")
    ax_t.set_title(f"runtime to the same accuracy "
                   f"($\\|r\\|/\\|b\\| \\leq$ {args.tol:g})", fontsize=11.5,
                   color="#222")
    ax_t.set_xlabel("unknowns $n$", fontsize=10)
    ax_t.set_ylabel("wall clock (ms)", fontsize=10)
    ax_t.grid(alpha=0.25, which="both")
    ax_t.legend(fontsize=8.5, loc="upper left", framealpha=0.92, borderpad=0.35)
    big = rows[-1]
    ax_t.text(0.97, 0.03,
              f"slopes fitted over the last four sizes; $\\Sigma_m$ is $O(n^3)$ "
              f"asymptotically\n"
              f"memory at $n = {big['n']}$:  $\\Sigma_m$ "
              f"{big['bytes_cov'] / 1e6:.0f} MB   vs   $A$ "
              f"{big['bytes_A'] / 1e3:.0f} kB\n"
              f"at $n = 10^6$:  $\\Sigma_m$ would be "
              f"{8.0 * (1e6 ** 2) / 1e12:.0f} TB",
              transform=ax_t.transAxes, fontsize=9.0, color=PAL["navy"],
              ha="right", va="bottom")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    plt.close(fig)
    return args.out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="8,12,16,20,24,32,40,48",
                   help="tiles per side for the runtime panel (n = m^2 each)")
    p.add_argument("--acc-size", type=int, default=24,
                   help="tiles per side for the accuracy panel")
    p.add_argument("--acc-iters", type=int, default=60,
                   help="most iterations to plot on the accuracy panel")
    p.add_argument("--screening", type=float, default=0.4,
                   help="c: coupling to the coolant (default: 0.4)")
    p.add_argument("--tol", type=float, default=1e-8,
                   help="relative residual that fixes the common iteration count")
    p.add_argument("--repeats", type=int, default=3,
                   help="timing repeats; the best of them is reported")
    p.add_argument("--min-time", type=float, default=0.03,
                   help="seconds each timing batch must take, so that fast calls are "
                        "not measuring the clock")
    p.add_argument("--out", default="figures/cg-vs-bayescg.png")
    args = p.parse_args()
    args.sizes = [int(s) for s in args.sizes.split(",")]

    print("timings (best of %d):" % args.repeats)
    rows = timings(args)
    acc = accuracy(args)
    print(f"accuracy on n = {acc['n']}, {acc['K']} iterations:")
    print(f"  max |BayesCG mean - CG iterate| / |CG iterate| = {acc['gap'].max():.2e}")
    print(f"  final error   CG {acc['cg'][-1]:.3e}   "
          f"Sigma_0 = A^-1 {acc['ainv'][-1]:.3e}   "
          f"Sigma_0 = I {acc['identity'][-1]:.3e}")
    print(f"  BayesCG's own estimate at the same point: {acc['claim'][-1]:.3e}  "
          f"(it is sqrt((n-m)/n), whatever the error does)")
    print("wrote", draw(acc, rows, args))


if __name__ == "__main__":
    main()
