# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""Conjugate gradients as a sequence of line searches.

ProbNum 2026 tutorial, §2. Third of the sweep animations, after
`jacobi-sweep-animation.py` and `gauss-seidel-sweep-animation.py`.

Solving $Ax=b$ for symmetric positive definite $A$ is minimising
$f(x) = \\tfrac12 x^\\top A x - b^\\top x$, whose gradient is $-r$. Each step is a
line search: pick a direction $p$, slide along $x + \\alpha p$ to the bottom of the
parabola $\\varphi(\\alpha) = f(x+\\alpha p)$, at $\\alpha^\\ast = p^\\top r / p^\\top A p$.

Steepest descent picks $p = r$ and zig-zags: the exact search leaves
$r_{k+1} \\perp p_k$, so each direction is orthogonal to the last one and undoes
progress already paid for. Conjugate gradients picks $p_{k+1} = r_{k+1} + \\beta_k p_k$
with the one $\\beta_k$ making $p_{k+1}^\\top A p_k = 0$ — orthogonality in the geometry
$A$ defines, which is exactly the condition for the new direction not to spoil the
minimisation already done along the old one. So the searches never have to be
repeated: $n$ of them solve an $n$-dimensional problem exactly, and in the plane
drawn here the second direction points straight at the solution.

The plane
---------
The operator is real: $f$ restricted to the plane spanned by the eigenvectors of
$A$'s smallest and largest eigenvalues, so the ellipse is as elongated as the die
is ill-conditioned ($\\kappa$ of the restriction equals $\\kappa$ of the full 64-unknown
problem). `--rotate` only chooses a basis within that plane.

The right-hand side inside the plane is *chosen*, not restricted, and that has to
be said. Restricting the die's own $b$ leaves an initial error that is essentially
the slow eigenvector alone — a smooth workload has almost no checkerboard content
— and steepest descent then finishes in one step. `--mode-mix` sets the split; the
default is the worst case for steepest descent, $\\sqrt{\\kappa}/(1+\\sqrt{\\kappa})$,
which drives its measured rate onto the bound $(\\kappa-1)/(\\kappa+1)$. Both numbers
are printed, along with the two identities the method rests on, so the claims can
be checked rather than believed.

Usage
-----
    uv run python/conjugate-gradients-animation.py                 # -> figures/conjugate-gradients.gif
    uv run python/conjugate-gradients-animation.py --sd-steps 60   # let steepest descent grind longer
    uv run python/conjugate-gradients-animation.py --format mp4    # needs ffmpeg

Without uv:  pip install numpy matplotlib pillow  &&  python python/conjugate-gradients-animation.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import FancyArrowPatch

PAL = dict(blue="#2a78d6", red="#C81919", navy="#101073", gray="#898781",
           orange="#eb6834", white="#fcfcfb", green="#008300")


# ----------------------------------------------------------------- the die problem
def grid_matrix(m: int, c: float) -> np.ndarray:
    """A = (c - Delta) on an m x m tiling, unknowns numbered row by row: k = i*m + j."""
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


def worst_plane(A: np.ndarray, rotate_deg: float, mix: float | None):
    """f restricted to the plane of A's slowest and fastest modes.

    Returns (A2, b2, kappa, mix). Restricting a quadratic to a plane through the
    origin leaves a quadratic in two variables, so the contours drawn from A2 are
    the true contours of f on that plane. The plane is spanned by the extreme
    eigenvectors, so cond(A2) = cond(A) exactly. See the module docstring on why
    b2 is chosen rather than restricted; mix = None takes the worst case for
    steepest descent, sqrt(kappa)/(1 + sqrt(kappa)).
    """
    w, V = np.linalg.eigh(A)
    Q = np.column_stack([V[:, 0], V[:, -1]])
    th = np.deg2rad(rotate_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = Q @ R                       # still an orthonormal basis of the same plane
    A2 = Q.T @ A @ Q
    w2, U2 = np.linalg.eigh(A2)
    if mix is None:
        sk = np.sqrt(w2[1] / w2[0])
        mix = float(sk / (1.0 + sk))
    x_star = U2 @ np.array([mix / np.sqrt(w2[0]), (1.0 - mix) / np.sqrt(w2[1])])
    x_star = x_star / np.linalg.norm(x_star)
    return A2, A2 @ x_star, float(w[-1] / w[0]), float(mix)


# ----------------------------------------------------------------- the methods
def cg_steps(A, b, iters: int, tol: float = 1e-14):
    """Conjugate gradients, recording what each line search needs in order to be drawn.

    alpha is computed in its line-search form, p^T r / p^T A p, rather than as
    r^T r / p^T A p: the two are equal, but only the first is visibly the minimiser
    of a parabola.
    """
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    bn = float(np.linalg.norm(b))
    steps = []
    for _ in range(iters):
        Ap = A @ p
        pAp = float(p @ Ap)
        pr = float(p @ r)
        alpha = pr / pAp
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        beta = float(r_new @ r_new) / float(r @ r)
        p_new = r_new + beta * p
        steps.append(dict(
            x=x.copy(), r=r.copy(), p=p.copy(), alpha=alpha, pAp=pAp, pr=pr,
            f0=0.5 * float(x @ (A @ x)) - float(b @ x),
            x_new=x_new.copy(), r_new=r_new.copy(), beta=beta,
            ortho=float(r_new @ p),               # exact line search: this is 0
            conj=float(p_new @ Ap)))              # conjugacy: this is 0 too
        x, r, p = x_new, r_new, p_new
        if np.linalg.norm(r) <= tol * bn:
            break
    return steps, x


def sd_path(A, b, iters: int):
    """Steepest descent with the same exact line search: p = r, alpha = r^T r / r^T A r."""
    x = np.zeros_like(b)
    xs = [x.copy()]
    for _ in range(iters):
        r = b - A @ x
        rAr = float(r @ (A @ r))
        if rAr <= 0.0:
            break
        x = x + (float(r @ r) / rAr) * r
        xs.append(x.copy())
    return xs


def a_norm_errors(A, xs, x_star):
    """||x - x*||_A / ||x*||_A: the quantity CG minimises. Printed, not drawn."""
    nrm = lambda e: float(np.sqrt(max(e @ (A @ e), 0.0)))
    ref = nrm(x_star)
    return np.array([max(nrm(x - x_star) / ref, 1e-16) for x in xs])


# ----------------------------------------------------------------- the trajectory
def build(args):
    A = grid_matrix(args.m, args.screening)
    A2, b2, kappa, mix = worst_plane(A, args.rotate, args.mode_mix)

    steps, x_end = cg_steps(A2, b2, iters=2)
    x_star = np.linalg.solve(A2, b2)
    sd = sd_path(A2, b2, args.sd_steps)
    cg_err = a_norm_errors(A2, [s["x"] for s in steps] + [x_end], x_star)
    sd_err = a_norm_errors(A2, sd, x_star)

    # the rate steepest descent actually achieves, over its last ten steps
    ks = len(sd_err) - 1
    sd_rate = float((sd_err[ks] / sd_err[max(ks - 10, 0)]) ** (1.0 / min(10, ks))) \
        if ks else float("nan")

    # one frame per beat: form the direction, slide along it, land on the minimum
    frames = []
    for k, st in enumerate(steps):
        common = dict(k=k, st=st, beta_in=steps[k - 1]["beta"] if k else None,
                      conj_in=steps[k - 1]["conj"] if k else None,
                      cg_done=k, sd_done=min(k, len(sd) - 1))
        for _ in range(args.dir_frames):
            frames.append(dict(phase="direction", alpha=0.0, **common))
        for i in range(1, args.search_frames + 1):
            frames.append(dict(phase="search", alpha=st["alpha"] * i / args.search_frames,
                               **common))
        for _ in range(args.land_frames):
            frames.append(dict(phase="land", alpha=st["alpha"],
                               **{**common, "cg_done": k + 1,
                                  "sd_done": min(k + 1, len(sd) - 1)}))
    for s in range(len(steps), len(sd)):
        frames.append(dict(phase="tail", alpha=None, k=len(steps) - 1, st=steps[-1],
                           beta_in=None, conj_in=None, cg_done=len(steps), sd_done=s))
    for _ in range(args.hold):
        frames.append(dict(phase="tail", alpha=None, k=len(steps) - 1, st=steps[-1],
                           beta_in=None, conj_in=None, cg_done=len(steps),
                           sd_done=len(sd) - 1))

    return dict(frames=frames, A=A, A2=A2, b2=b2, kappa=kappa, mix=mix, steps=steps,
                x_star=x_star, sd=sd, cg_err=cg_err, sd_err=sd_err, sd_rate=sd_rate)


# ----------------------------------------------------------------- drawing
def plane_panel(ax, sim):
    """Contours of f on the plane, plus the exact minimiser."""
    A2, b2, xs = sim["A2"], sim["b2"], sim["x_star"]
    pts = np.array([[0.0, 0.0], xs] + [list(p) for p in sim["sd"]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.22 * max(hi - lo).item() + 1e-9
    lo, hi = lo - pad, hi + pad
    XX, YY = np.meshgrid(np.linspace(lo[0], hi[0], 320),
                         np.linspace(lo[1], hi[1], 320))
    F = (0.5 * (A2[0, 0] * XX ** 2 + 2 * A2[0, 1] * XX * YY + A2[1, 1] * YY ** 2)
         - b2[0] * XX - b2[1] * YY)
    fmin = 0.5 * float(xs @ (A2 @ xs)) - float(b2 @ xs)
    # geometric in height above the bottom, out to the corner of the window, so the
    # ellipses fill the panel; solid, or contour would dash every negative level
    ax.contour(XX, YY, F, levels=fmin + (F.max() - fmin) * np.geomspace(0.004, 1.0, 15),
               colors="#c3d2e6", linewidths=0.9, linestyles="solid", zorder=1)
    ax.plot([xs[0]], [xs[1]], marker="*", ms=15, color=PAL["red"], zorder=8,
            label="the solution")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$f$ on the die's worst plane   ($\\kappa = {sim['kappa']:.1f}$)",
                 fontsize=11.5, color="#222")


def right_angle(vertex, u, v, size):
    """Three points tracing the little square that marks u perpendicular to v."""
    u = u / (np.linalg.norm(u) + 1e-30)
    v = v / (np.linalg.norm(v) + 1e-30)
    a, b, c = vertex + size * u, vertex + size * (u + v), vertex + size * v
    return [a[0], b[0], c[0]], [a[1], b[1], c[1]]


def animate(sim, args):
    frames, steps = sim["frames"], sim["steps"]

    fig, (ax_pl, ax_ls) = plt.subplots(1, 2, figsize=(9.4, 4.6),
                                       gridspec_kw=dict(width_ratios=[1.0, 0.95]))
    fig.patch.set_facecolor("white")

    # ---------------- the plane
    plane_panel(ax_pl, sim)
    span = max(ax_pl.get_xlim()[1] - ax_pl.get_xlim()[0],
               ax_pl.get_ylim()[1] - ax_pl.get_ylim()[0])
    (sd_line,) = ax_pl.plot([], [], "-o", color=PAL["gray"], lw=1.3, ms=3.4,
                            alpha=0.85, zorder=3, label="steepest descent")
    (cg_line,) = ax_pl.plot([], [], "-o", color=PAL["blue"], lw=2.4, ms=6.0,
                            zorder=4, label="conjugate gradients")
    (cg_head,) = ax_pl.plot([], [], "o", color=PAL["blue"], ms=9.5, mec="white",
                            mew=1.4, zorder=7)
    (search_line,) = ax_pl.plot([], [], ":", color=PAL["navy"], lw=1.4, alpha=0.8,
                                zorder=2)
    arr = dict(arrowstyle="-|>", shrinkA=0.0, shrinkB=0.0, mutation_scale=15)
    p_arrow = FancyArrowPatch((0, 0), (0, 0), color=PAL["navy"], lw=1.7,
                              visible=False, zorder=5, **arr)
    r_arrow = FancyArrowPatch((0, 0), (0, 0), color=PAL["gray"], lw=1.8,
                              linestyle="dashed", visible=False, zorder=5, **arr)
    pnew_arrow = FancyArrowPatch((0, 0), (0, 0), color=PAL["green"], lw=2.2,
                                 visible=False, zorder=5, **arr)
    for a in (p_arrow, r_arrow, pnew_arrow):
        ax_pl.add_patch(a)
    (perp,) = ax_pl.plot([], [], "-", color=PAL["navy"], lw=1.4, zorder=6)
    note = ax_pl.text(0.03, 0.035, "", transform=ax_pl.transAxes, fontsize=9.5,
                      color=PAL["green"], va="bottom", ha="left", zorder=9,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                                alpha=0.85))
    ax_pl.legend(loc="upper right", fontsize=9, framealpha=0.85, borderpad=0.3,
                 handlelength=1.6)

    # ---------------- the line search
    (para,) = ax_ls.plot([], [], "-", color=PAL["navy"], lw=2.0)
    (tang,) = ax_ls.plot([], [], "-", color=PAL["orange"], lw=1.8)
    (dot,) = ax_ls.plot([], [], "o", color=PAL["blue"], ms=9.0, mec="white", mew=1.3,
                        zorder=5)
    (vert,) = ax_ls.plot([], [], "--", color=PAL["red"], lw=1.3)
    (star,) = ax_ls.plot([], [], "*", color=PAL["red"], ms=14, zorder=4)
    ax_ls.set_title(r"the line search:  $\varphi(\alpha) = f(x + \alpha p)$",
                    fontsize=11.5, color="#222")
    ax_ls.set_xlabel(r"$\alpha$", fontsize=10)
    ax_ls.grid(alpha=0.25)

    head = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=12.0, color="#222")
    calc = fig.text(0.5, 0.045, "", ha="center", va="center", fontsize=11.5,
                    color="#222")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.925))

    def draw(fi):
        f = frames[fi]
        st, k = f["st"], f["k"]
        x, p = st["x"], st["p"]

        sd_pts = np.array(sim["sd"][: f["sd_done"] + 1])
        sd_line.set_data(sd_pts[:, 0], sd_pts[:, 1])
        cg_pts = [s["x"] for s in steps[: f["cg_done"]]]
        if f["phase"] in ("land", "tail"):
            cg_pts = cg_pts + [steps[f["cg_done"] - 1]["x_new"]]
        elif f["alpha"] is not None:
            cg_pts = cg_pts + [x + f["alpha"] * p]
        cg_pts = np.array(cg_pts)
        cg_line.set_data(cg_pts[:, 0], cg_pts[:, 1])
        cg_head.set_data([cg_pts[-1, 0]], [cg_pts[-1, 1]])

        for a in (p_arrow, r_arrow, pnew_arrow):
            a.set_visible(False)
        perp.set_data([], [])
        search_line.set_data([], [])
        note.set_text("")

        if f["phase"] in ("direction", "search"):
            tip = x + st["alpha"] * p
            p_arrow.set_positions(tuple(x), tuple(tip)); p_arrow.set_visible(True)
            # the line right across the bowl: every point the search is choosing between
            t = 3.0 * span / (np.linalg.norm(p) + 1e-30)
            ends = np.array([x - t * p, x + t * p])
            search_line.set_data(ends[:, 0], ends[:, 1])
        if f["phase"] == "direction" and k > 0:
            # the two candidate steps, each drawn to where its own exact line search
            # lands: down the residual, or down r + beta p
            r_k = st["r"]
            a_sd = float(r_k @ r_k) / float(r_k @ (sim["A2"] @ r_k))
            r_arrow.set_positions(tuple(x), tuple(x + a_sd * r_k))
            pnew_arrow.set_positions(tuple(x), tuple(x + st["alpha"] * p))
            r_arrow.set_visible(True); pnew_arrow.set_visible(True)
            p_arrow.set_visible(False)
            note.set_text("green: $r + \\beta p$ — it lands on the solution")
        if f["phase"] == "land":
            perp.set_data(*right_angle(st["x_new"], st["p"], st["r_new"], 0.035 * span))

        if f["alpha"] is not None:
            a_star = st["alpha"]
            aa = np.linspace(-0.35 * a_star, 1.75 * a_star, 200)
            phi = st["f0"] - aa * st["pr"] + 0.5 * aa ** 2 * st["pAp"]
            para.set_data(aa, phi)
            cur = f["alpha"]
            phi_cur = st["f0"] - cur * st["pr"] + 0.5 * cur ** 2 * st["pAp"]
            slope = -st["pr"] + cur * st["pAp"]
            dt = 0.42 * a_star
            tang.set_data([cur - dt, cur + dt],
                          [phi_cur - slope * dt, phi_cur + slope * dt])
            dot.set_data([cur], [phi_cur])
            phi_min = st["f0"] - a_star * st["pr"] + 0.5 * a_star ** 2 * st["pAp"]
            star.set_data([a_star], [phi_min])
            vert.set_data([a_star, a_star], [phi_min, phi.max()])
            ax_ls.set_xlim(aa[0], aa[-1])
            marg = 0.12 * (phi.max() - phi_min) + 1e-12
            ax_ls.set_ylim(phi_min - marg, phi.max() + marg)

        if f["phase"] == "direction":
            if k == 0:
                head.set_text("step 1:  the steepest direction, $p_1 = r_0$")
                calc.set_text("with $x_0 = 0$, CG and steepest descent take the same "
                              "first step")
            else:
                head.set_text(f"step {k + 1}:  $p_{{{k + 1}}} = r_{k + 1} + "
                              f"\\beta_{k} p_{{{k}}}$,  the residual corrected")
                calc.set_text(f"$\\beta_{k} = {f['beta_in']:.3f}$,   and the point of "
                              f"it:  $p_{{{k + 1}}}^\\top A\\, p_{{{k}}} = "
                              f"{f['conj_in']:.0e}$")
        elif f["phase"] == "search":
            head.set_text(f"step {k + 1}:  exact line search along $p_{{{k + 1}}}$")
            calc.set_text(f"$\\alpha^\\ast = p^\\top r\\,/\\,p^\\top\\! A p = "
                          f"{st['alpha']:.3f}$,   at $\\alpha = {f['alpha']:.3f}$")
        elif f["phase"] == "land":
            head.set_text(f"step {k + 1} done:  $r_{{{k + 1}}} \\perp p_{{{k + 1}}}$")
            calc.set_text(f"$r_{{{k + 1}}}^\\top p_{{{k + 1}}} = {st['ortho']:.0e}$ — "
                          f"nothing left to gain along that line")
        else:
            head.set_text(f"CG: done in {len(steps)} steps, $n = 2$   —   "
                          f"steepest descent: step {f['sd_done']}")
            calc.set_text(f"CG's rate is set by $\\sqrt{{\\kappa}} = "
                          f"{np.sqrt(sim['kappa']):.1f}$, steepest descent's by "
                          f"$\\kappa = {sim['kappa']:.1f}$")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    written = save(anim, fig, args, "conjugate-gradients")
    plt.close(fig)
    return written


# ----------------------------------------------------------------- writing it out
def even_pixels(fig, dpi: float) -> None:
    """Nudge the figure up to an even pixel count in both directions (H.264 4:2:0)."""
    fig.set_dpi(dpi)
    w, h = fig.canvas.get_width_height()
    if w % 2 or h % 2:
        fig.set_size_inches((w + w % 2 + 0.5) / dpi, (h + h % 2 + 0.5) / dpi)


def save(anim, fig, args, stem: str) -> list[str]:
    """Write the animation in every requested format; return the files written."""
    formats = ("gif", "mp4") if args.format == "both" else (args.format,)
    if "mp4" in formats and not FFMpegWriter.isAvailable():
        raise SystemExit("mp4 output needs ffmpeg on the PATH "
                         "(apt install ffmpeg / brew install ffmpeg); "
                         "or use --format gif.")
    written = []
    for fmt in formats:
        if args.out and len(formats) == 1:
            out = args.out
        elif args.out:
            out = f"{args.out.rsplit('.', 1)[0]}.{fmt}"
        else:
            out = os.path.join(args.outdir, f"{stem}.{fmt}")
        if os.path.dirname(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
        if fmt == "gif":
            writer = PillowWriter(fps=args.fps)
        else:
            even_pixels(fig, args.dpi)
            writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=args.bitrate,
                                  extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high",
                                              "-preset", "slow", "-movflags", "+faststart"])
        anim.save(out, writer=writer, dpi=args.dpi)
        written.append(out)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=8, help="tiles per side of the die")
    p.add_argument("--screening", type=float, default=0.4,
                   help="c: coupling to the coolant (default: 0.4)")
    p.add_argument("--rotate", type=float, default=25.0,
                   help="rotate the drawing axes within the plane, in degrees; a choice "
                        "of basis only, so that the ellipses are not axis-aligned")
    p.add_argument("--mode-mix", type=float, default=None,
                   help="how the initial error splits between the die's slow and fast "
                        "modes; default is the worst case for steepest descent, "
                        "sqrt(kappa)/(1+sqrt(kappa)). Near 0 or 1 the error is a single "
                        "mode and steepest descent converges in one step")
    p.add_argument("--sd-steps", type=int, default=40,
                   help="steepest-descent steps to draw (it does not finish)")
    p.add_argument("--dir-frames", type=int, default=5,
                   help="frames spent showing how the direction was formed")
    p.add_argument("--search-frames", type=int, default=10,
                   help="frames spent sliding along the line")
    p.add_argument("--land-frames", type=int, default=7,
                   help="frames spent on the landing point and its right angle")
    p.add_argument("--hold", type=int, default=12, help="frames to hold at the end")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--format", choices=("gif", "mp4", "both"), default="gif",
                   help="'mp4' needs ffmpeg on the PATH")
    p.add_argument("--bitrate", type=int, default=2400, help="mp4 bitrate in kbit/s")
    p.add_argument("--outdir", default="figures", help="directory for default names")
    p.add_argument("--out", default=None, help="output file (overrides --outdir)")
    args = p.parse_args()

    sim = build(args)
    written = animate(sim, args)
    st = sim["steps"]
    kappa = sim["kappa"]
    print(f"{', '.join(written)}   die {args.m}x{args.m} -> worst 2-D plane, "
          f"kappa = {kappa:.3f}, sqrt(kappa) = {np.sqrt(kappa):.3f}, "
          f"{len(sim['frames'])} frames")
    print(f"  mode mix {sim['mix']:.3f}  ->  steepest descent's measured rate "
          f"{sim['sd_rate']:.4f} against its bound "
          f"(kappa-1)/(kappa+1) = {(kappa - 1) / (kappa + 1):.4f}")
    print(f"  CG: {len(st)} steps, final ||x-x*||_A/||x*||_A = {sim['cg_err'][-1]:.2e}")
    print(f"  steepest descent, {len(sim['sd']) - 1} steps: {sim['sd_err'][-1]:.2e}")
    print("  verifications  max |r_{k+1}^T p_k| = "
          f"{max(abs(s['ortho']) for s in st):.2e}   (exact line search)")
    print("                 max |p_{k+1}^T A p_k| = "
          f"{max(abs(s['conj']) for s in st):.2e}   (A-conjugacy)")


if __name__ == "__main__":
    main()
