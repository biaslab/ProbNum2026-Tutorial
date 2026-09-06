# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""BayesCG: the same two steps as conjugate gradients, as a Bayesian update.

ProbNum 2026 tutorial, §3. Companion to `conjugate-gradients-animation.py` — same
die, same plane, same trajectory — so the two can be played back to back and the
only new thing on screen is the uncertainty.

BayesCG (Cockayne, Oates, Ipsen & Girolami, *Bayesian Analysis* 14(3), 2019; the
paper is in `literature/`) puts a prior on the solution, $x \\sim N(x_0, \\Sigma_0)$,
and treats each search direction $s_i$ as a *noiseless observation* of it:

    y_i = s_i^T b = s_i^T A x .

That is a linear-Gaussian observation, so the posterior is closed-form. With
$S = [s_1,\\dots,s_m]$ and $\\Lambda = S^T A \\Sigma_0 A S$,

    x_m = x_0 + \\Sigma_0 A S \\Lambda^{-1} S^T r_0 ,
    \\Sigma_m = \\Sigma_0 - \\Sigma_0 A S \\Lambda^{-1} S^T A \\Sigma_0 .

Two things the animation is built to show:

* **The update is a collapse, not a shrink.** Each observation is exact, so it
  removes one dimension of uncertainty outright: the ellipse flattens onto the
  line $\\{x : s^T A x = s^T b\\}$ and keeps no width across it. In two dimensions
  two observations leave a point, and that point is the solution. The right-hand
  panel is the same event in one dimension — the belief about the quantity being
  observed, narrowing onto the value observed.
* **With $\\Sigma_0 = A^{-1}$ the posterior mean is exactly the CG iterate.** The blue
  path here is the blue path of the CG animation, drawn by a different argument;
  `--prior identity` breaks that and moves the path, which is the quickest way to
  see that the prior is a modelling choice and not a formality.

What it costs, and what it knows
--------------------------------
$\\Sigma_m$ is a dense $n \\times n$ object; the die has $n = 64$ here and a real one
has millions. And the belief's own error estimate is
$\\mathbb{E}[\\|x - x_m\\|_A^2] = \\mathrm{tr}(A\\Sigma_m) = n - m$ exactly — for any
$b$, any right-hand side, any directions. It counts directions, not data: after
one observation it reports "half the error left" whatever the actual error did.
`main` prints $\\mathrm{tr}(A\\Sigma_m)$ against $n - m$ and against the true error,
so the gap can be read rather than argued about. That is the point the tutorial
needs: a posterior is not automatically an error bar.

The plane
---------
As in the CG animation: $f$ restricted to the plane of $A$'s slowest and fastest
modes, so $\\kappa$ is the die's own; the right-hand side inside the plane is chosen
so the problem is not degenerate (see that script's docstring). The intermediate
frames of each update are honest posteriors, not a graphical tween: the
observation is faded in by giving it a noise variance that decays to zero, and
BayesCG's exact update is the limit.

Usage
-----
    uv run python/bayescg-animation.py                     # -> figures/bayescg.gif
    uv run python/bayescg-animation.py --prior identity    # the wrong prior: the mean leaves CG's path
    uv run python/bayescg-animation.py --samples 0         # ellipses only, no sample cloud
    uv run python/bayescg-animation.py --format mp4        # needs ffmpeg

Without uv:  pip install numpy matplotlib pillow  &&  python python/bayescg-animation.py
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
    """f restricted to the plane of A's slowest and fastest modes; see the CG script."""
    w, V = np.linalg.eigh(A)
    Q = np.column_stack([V[:, 0], V[:, -1]])
    th = np.deg2rad(rotate_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = Q @ R
    A2 = Q.T @ A @ Q
    w2, U2 = np.linalg.eigh(A2)
    if mix is None:
        sk = np.sqrt(w2[1] / w2[0])
        mix = float(sk / (1.0 + sk))
    x_star = U2 @ np.array([mix / np.sqrt(w2[0]), (1.0 - mix) / np.sqrt(w2[1])])
    x_star = x_star / np.linalg.norm(x_star)
    return A2, A2 @ x_star, float(w[-1] / w[0])


def cg_directions(A, b, iters: int):
    """CG's search directions, which are also BayesCG's most informative ones."""
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    dirs = []
    for _ in range(iters):
        Ap = A @ p
        alpha = float(p @ r) / float(p @ Ap)
        dirs.append(p.copy())
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = float(r_new @ r_new) / float(r @ r)
        p = r_new + beta * p
        r = r_new
    return dirs, x


# ----------------------------------------------------------------- the update
def posterior(A, b, Sig0, S, noise):
    """BayesCG's posterior from m observations, the i-th given variance noise[i].

    noise = 0 is the method as published: exact observations of s_i^T A x. A large
    variance on the newest observation is how the animation fades it in -- every
    intermediate frame is the honest posterior for a noisier version of the same
    observation, and BayesCG's update is the limit as that noise goes to zero.
    """
    x0 = np.zeros(A.shape[0])
    if S.shape[1] == 0:
        return x0, Sig0.copy()
    K = Sig0 @ A @ S                                   # Sigma_0 A S
    Lam = S.T @ A @ Sig0 @ A @ S + np.diag(np.atleast_1d(noise))
    x_m = x0 + K @ np.linalg.solve(Lam, S.T @ (b - A @ x0))
    Sig_m = Sig0 - K @ np.linalg.solve(Lam, K.T)
    return x_m, 0.5 * (Sig_m + Sig_m.T)                # symmetrise off round-off


def sym_sqrt(S):
    """The symmetric square root, which stays continuous as S collapses."""
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def ellipse(mu, Sig, n=200):
    """The 1-sigma contour. Degenerate covariance gives a segment, which is the point."""
    th = np.linspace(0.0, 2.0 * np.pi, n)
    pts = mu[:, None] + sym_sqrt(Sig) @ np.vstack([np.cos(th), np.sin(th)])
    return pts[0], pts[1]


# ----------------------------------------------------------------- the trajectory
def build(args):
    A = grid_matrix(args.m, args.screening)
    A2, b2, kappa = worst_plane(A, args.rotate, args.mode_mix)
    x_star = np.linalg.solve(A2, b2)
    dirs, cg_end = cg_directions(A2, b2, 2)
    Sig0 = np.linalg.inv(A2) if args.prior == "ainv" else np.eye(2)

    means, covs = [], []                       # the exact posteriors, 0..m observations
    for m in range(len(dirs) + 1):
        S = np.column_stack(dirs[:m]) if m else np.zeros((2, 0))
        xm, Sm = posterior(A2, b2, Sig0, S, np.zeros(m))
        means.append(xm); covs.append(Sm)

    z = np.random.default_rng(args.seed).standard_normal((2, args.samples))

    frames = []
    for m in range(1, len(dirs) + 1):
        S = np.column_stack(dirs[:m])
        s = dirs[m - 1]
        mu_pre, Sig_pre = means[m - 1], covs[m - 1]
        # what the belief expects to see, and what is actually observed
        pred_mu = float(s @ (A2 @ mu_pre))
        pred_var = float(s @ (A2 @ Sig_pre @ A2 @ s))
        obs = float(s @ b2)
        common = dict(m=m, s=s, S=S, pred_mu=pred_mu, pred_var=pred_var, obs=obs)

        for _ in range(args.propose_frames):
            frames.append(dict(phase="propose", mu=mu_pre, Sig=Sig_pre,
                               var=pred_var, done=m - 1, **common))
        for i in range(1, args.fade_frames + 1):
            # Noise on the newest observation only; earlier ones stay exact. The level
            # is picked so that the belief's *standard deviation* across the new
            # constraint shrinks linearly to zero -- with variance ratio rho, a noise
            # of pred_var*rho/(1-rho) leaves exactly rho of the variance -- which
            # makes the collapse look even instead of happening in the first frame.
            rho = (1.0 - i / args.fade_frames) ** 2
            nz = np.zeros(m)
            nz[-1] = pred_var * rho / (1.0 - rho) if rho < 1.0 else np.inf
            mu_i, Sig_i = posterior(A2, b2, Sig0, S, nz)
            frames.append(dict(phase="condition", mu=mu_i, Sig=Sig_i,
                               var=float(s @ (A2 @ Sig_i @ A2 @ s)),
                               done=m - 1, **common))
        for _ in range(args.post_frames):
            frames.append(dict(phase="posterior", mu=means[m], Sig=covs[m],
                               var=float(s @ (A2 @ covs[m] @ A2 @ s)),
                               done=m, **common))
    for _ in range(args.hold):
        # keep the last step's predictive variance: it only sets the panel's window,
        # and a zero would collapse the axis onto a single value
        frames.append(dict(phase="end", mu=means[-1], Sig=covs[-1], var=0.0,
                           done=len(dirs), m=len(dirs), s=dirs[-1],
                           S=np.column_stack(dirs), pred_mu=frames[-1]["pred_mu"],
                           pred_var=frames[-1]["pred_var"], obs=frames[-1]["obs"]))

    # what the belief claims about its own error, against what the error is
    trace = [float(np.trace(A2 @ S)) for S in covs]
    err = [float(np.sqrt((m_ - x_star) @ A2 @ (m_ - x_star))) for m_ in means]
    return dict(frames=frames, A=A, A2=A2, b2=b2, kappa=kappa, Sig0=Sig0, z=z,
                means=means, covs=covs, x_star=x_star, dirs=dirs, cg_end=cg_end,
                trace=trace, err=err)


# ----------------------------------------------------------------- drawing
def animate(sim, args):
    frames = sim["frames"]
    n = 2

    fig, (ax_pl, ax_ob) = plt.subplots(1, 2, figsize=(9.4, 4.6),
                                       gridspec_kw=dict(width_ratios=[1.0, 0.95]))
    fig.patch.set_facecolor("white")

    # ---------------- the plane: the belief about x
    px, py = ellipse(sim["means"][0], sim["covs"][0])
    pts = np.array([[px.min(), py.min()], [px.max(), py.max()],
                    sim["x_star"]] + [list(m_) for m_ in sim["means"]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.18 * max(hi - lo).item() + 1e-9
    lo, hi = lo - pad, hi + pad
    ax_pl.plot(px, py, "--", color=PAL["gray"], lw=1.2, alpha=0.9, label="the prior")
    (band,) = ax_pl.plot([], [], "-", color=PAL["blue"], lw=2.0, zorder=5,
                         label="the belief now")
    fill = ax_pl.fill([], [], color=PAL["blue"], alpha=0.13, zorder=2)[0]
    (cloud,) = ax_pl.plot([], [], "o", color=PAL["navy"], ms=2.8, alpha=0.5, zorder=4)
    (line,) = ax_pl.plot([], [], ":", color=PAL["navy"], lw=1.4, alpha=0.85, zorder=3)
    (path,) = ax_pl.plot([], [], "-o", color=PAL["blue"], lw=2.2, ms=5.5, zorder=6,
                         label="the posterior mean")
    (head,) = ax_pl.plot([], [], "o", color=PAL["blue"], ms=9.0, mec="white", mew=1.3,
                         zorder=7)
    ax_pl.plot([sim["x_star"][0]], [sim["x_star"][1]], marker="*", ms=15,
               color=PAL["red"], zorder=8, label="the solution")
    ax_pl.set_xlim(lo[0], hi[0]); ax_pl.set_ylim(lo[1], hi[1])
    ax_pl.set_aspect("equal")
    ax_pl.set_xticks([]); ax_pl.set_yticks([])
    ax_pl.set_title("the belief about $x$", fontsize=11.5, color="#222")
    ax_pl.legend(loc="upper right", fontsize=8.5, framealpha=0.85, borderpad=0.3,
                 handlelength=1.6)
    note = ax_pl.text(0.03, 0.035, "", transform=ax_pl.transAxes, fontsize=9.5,
                      color=PAL["navy"], va="bottom", ha="left", zorder=9,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                                alpha=0.85))

    # ---------------- the same update, in the one dimension being observed
    (dens,) = ax_ob.plot([], [], "-", color=PAL["blue"], lw=2.2)
    dfill = ax_ob.fill([], [], color=PAL["blue"], alpha=0.15)[0]
    (obs_line,) = ax_ob.plot([], [], "--", color=PAL["red"], lw=1.6)
    ax_ob.set_title(r"the observation:  $s^\top\! A x = s^\top b$", fontsize=11.5,
                    color="#222")
    ax_ob.set_xlabel(r"$s^\top\! A x$", fontsize=10)
    ax_ob.set_yticks([])
    ax_ob.set_ylim(0.0, 1.12)
    ax_ob.grid(alpha=0.25)
    ob_note = ax_ob.text(0.035, 0.965, "", fontsize=9.5, color=PAL["red"], ha="left",
                         va="top", transform=ax_ob.transAxes)

    head_txt = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=12.0,
                        color="#222")
    calc = fig.text(0.5, 0.045, "", ha="center", va="center", fontsize=11.5,
                    color="#222")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.925))

    def draw(fi):
        f = frames[fi]
        mu, Sig, s, m = f["mu"], f["Sig"], f["s"], f["m"]

        ex, ey = ellipse(mu, Sig)
        band.set_data(ex, ey)
        fill.set_xy(np.column_stack([ex, ey]))
        if args.samples:
            pts = mu[:, None] + sym_sqrt(Sig) @ sim["z"]
            cloud.set_data(pts[0], pts[1])
        head.set_data([mu[0]], [mu[1]])
        done = np.array(sim["means"][: f["done"] + 1] + ([] if f["phase"] in
                        ("posterior", "end") else [mu]))
        path.set_data(done[:, 0], done[:, 1])

        # the observed constraint: {x : s^T A x = s^T b}, a line in the plane
        if f["phase"] in ("propose", "condition"):
            g = sim["A2"] @ s                       # normal of the line
            base = g * f["obs"] / float(g @ g)
            tang = np.array([-g[1], g[0]]) / np.linalg.norm(g)
            t = 3.0 * max(ax_pl.get_xlim()[1] - ax_pl.get_xlim()[0],
                          ax_pl.get_ylim()[1] - ax_pl.get_ylim()[0])
            ends = np.array([base - t * tang, base + t * tang])
            line.set_data(ends[:, 0], ends[:, 1])
        else:
            line.set_data([], [])

        note.set_text("" if f["phase"] != "propose" else
                      "dotted: every $x$ that agrees with\nthe observation about to be made")

        # the belief about the observed quantity, scaled to unit height so that the
        # collapse reads as narrowing rather than as a spike shooting off the panel.
        # The window is fixed for the whole step -- it comes from the *prior*
        # predictive -- so the narrowing is not confounded with a moving axis.
        mu_o = float(s @ (sim["A2"] @ mu))
        centre = 0.5 * (f["obs"] + f["pred_mu"])
        half = max(3.4 * np.sqrt(f["pred_var"]),
                   1.4 * abs(f["obs"] - f["pred_mu"]), 1e-9)
        span = 2.0 * half
        sd = max(np.sqrt(max(f["var"], 0.0)), 0.0035 * span)
        # a coarse grid for the window plus a fine one around the peak: once the
        # belief has collapsed, a uniform grid steps straight over the spike
        xs = np.unique(np.concatenate([np.linspace(centre - half, centre + half, 320),
                                       np.linspace(mu_o - 6 * sd, mu_o + 6 * sd, 220)]))
        d = np.exp(-0.5 * ((xs - mu_o) / sd) ** 2)
        dens.set_data(xs, d)
        dfill.set_xy(np.column_stack([np.r_[xs, xs[::-1]], np.r_[d, np.zeros_like(d)]]))
        obs_line.set_data([f["obs"], f["obs"]], [0.0, 1.08])
        ax_ob.set_xlim(centre - half, centre + half)
        ob_note.set_text(f"observed:  $s^\\top b = {f['obs']:.2f}$")

        if f["phase"] == "propose":
            head_txt.set_text(f"observation {m} of {n}:  direction $s_{m}$")
            calc.set_text(f"the belief expects $s^\\top\\! A x = {f['pred_mu']:.2f} \\pm "
                          f"{np.sqrt(f['pred_var']):.2f}$;   the answer is "
                          f"{f['obs']:.2f}")
        elif f["phase"] == "condition":
            head_txt.set_text("conditioning on it")
            calc.set_text("the observation is noiseless, so the width across the line "
                          "shrinks to nothing")
        elif f["phase"] == "posterior":
            head_txt.set_text(f"posterior after {m} observation"
                              f"{'s' if m > 1 else ''}")
            # the n - m identity is Sigma_0 = A^{-1} only; with any other prior the
            # trace is just a number, and claiming the identity would be false
            ident = " = n - m" if args.prior == "ainv" else ""
            calc.set_text(f"$\\|x_{m} - x^\\ast\\|_A = {sim['err'][m]:.2f}$,   and the "
                          f"belief's own estimate: $\\mathrm{{tr}}(A\\Sigma_{m}) = "
                          f"{sim['trace'][m]:.2f}{ident}$")
        else:
            head_txt.set_text(f"$n$ observations, $n = {n}$:  the posterior is the "
                              f"solution, with no variance left")
            calc.set_text(r"$\mathrm{tr}(A\Sigma_m) = n - m$ for any $b$ — the belief "
                          r"counts directions, not data"
                          if args.prior == "ainv" else
                          r"prior $\Sigma_0 = I$: the mean is not CG's iterate, and "
                          r"$\mathrm{tr}(A\Sigma_m) = n - m$ does not hold")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    written = save(anim, fig, args, "bayescg")
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
            suffix = "" if args.prior == "ainv" else f"-{args.prior}"
            out = os.path.join(args.outdir, f"{stem}{suffix}.{fmt}")
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
                   help="rotate the drawing axes within the plane, in degrees")
    p.add_argument("--mode-mix", type=float, default=None,
                   help="how the initial error splits between the die's slow and fast "
                        "modes; default is the worst case for steepest descent")
    p.add_argument("--prior", choices=("ainv", "identity"), default="ainv",
                   help="Sigma_0: 'ainv' is A^{-1}, for which the posterior mean is "
                        "exactly the CG iterate; 'identity' is the wrong prior, and "
                        "moves the mean off CG's path")
    p.add_argument("--samples", type=int, default=60,
                   help="draws from the belief to scatter (0 for none)")
    p.add_argument("--seed", type=int, default=0, help="seed for the sample cloud")
    p.add_argument("--propose-frames", type=int, default=7,
                   help="frames showing the direction and what it will reveal")
    p.add_argument("--fade-frames", type=int, default=14,
                   help="frames over which the observation's noise decays to zero")
    p.add_argument("--post-frames", type=int, default=7,
                   help="frames on the exact posterior")
    p.add_argument("--hold", type=int, default=14, help="frames to hold at the end")
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
    n = 2
    print(f"{', '.join(written)}   die {args.m}x{args.m} -> worst 2-D plane, "
          f"kappa = {sim['kappa']:.3f}, prior Sigma_0 = "
          f"{'A^-1' if args.prior == 'ainv' else 'I'}, {len(sim['frames'])} frames")
    gap = np.max(np.abs(sim["means"][-1] - sim["x_star"]))
    print(f"  posterior mean after {n} observations vs the exact solution: {gap:.2e}")
    if args.prior == "ainv":
        print("  posterior mean vs the CG iterate: "
              f"{np.max(np.abs(sim['means'][-1] - sim['cg_end'])):.2e}")
    print("  m   tr(A Sigma_m)     n - m*    ||x_m - x*||_A"
          "        (* the identity holds for Sigma_0 = A^-1 only)")
    for m in range(n + 1):
        print(f"  {m}   {sim['trace'][m]:13.6f}   {n - m:7d}     {sim['err'][m]:.4f}")


if __name__ == "__main__":
    main()
