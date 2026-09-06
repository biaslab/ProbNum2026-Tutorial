# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""Heat dissipating through the die — the transient whose fixed point is A x = b.

ProbNum 2026 tutorial, §1. The slide claim is that the control loop needs the
steady-state temperature of every tile *faster than the die's thermal time
constant*; this animation is that sentence, drawn.

The physics is the one on the slide. Tile i dissipates power b_i, conducts heat
to the four tiles it touches, and loses heat to the coolant in proportion to how
far above ambient it sits. Out of steady state the imbalance drives the
temperature:

    du/dt = b - A u,        A = (c - Delta) on the 5-point stencil,

so u(t) -> A^{-1} b, and the linear system of the tutorial is the fixed point of
the physical relaxation. Two remarks worth making out loud while it plays:

* Explicit Euler on this equation, u <- u + dt (b - A u), *is* the Richardson
  iteration -- a stationary iterative method of §2. The die is running a linear
  solver on itself, in real time, whether or not anyone asks it to. What the
  control loop wants is the answer sooner than the silicon can relax to it.
* The slowest mode decays with time constant tau = 1 / lambda_min(A), which is
  what the time axis is measured in. On the unscreened lattice (c = 0) tau grows
  with the die, which is the same statement as "the round count grows with the
  diameter" in §4.9 -- one correlation length, two vocabularies.

Usage
-----
    uv run python/heat-dissipation-animation.py                    # 24x24 die -> heat-dissipation.gif
    uv run python/heat-dissipation-animation.py --switch           # workload moves mid-flight
    uv run python/heat-dissipation-animation.py --screening 0.0    # poorly cooled: slow, wide
    uv run python/heat-dissipation-animation.py --format mp4       # needs ffmpeg
    uv run python/heat-dissipation-animation.py --format both      # gif + mp4 in one run

Without uv:  pip install numpy matplotlib pillow  &&  python heat-dissipation-animation.py
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

PAL = dict(blue="#2a78d6", red="#C81919", navy="#101073", gray="#898781",
           orange="#eb6834", white="#fcfcfb")


# ----------------------------------------------------------------- the operator
def apply_A(u: np.ndarray, c: float) -> np.ndarray:
    """A u for A = (c - Delta) on an m x m lattice, ambient (zero) outside the die.

    Matrix-free five-point stencil: exactly the `grid_matrix` of the notebook,
    applied to a field held as an (m, m) array instead of a flattened vector.
    """
    out = (4.0 + c) * u
    out[1:, :] -= u[:-1, :]
    out[:-1, :] -= u[1:, :]
    out[:, 1:] -= u[:, :-1]
    out[:, :-1] -= u[:, 1:]
    return out


def power_map(m: int, hotspots, width: float = 0.10) -> np.ndarray:
    """Dissipated power b: a few Gaussian workloads on the die, in (m, m) form."""
    g = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(g, g, indexing="ij")
    b = np.zeros((m, m))
    for cx, cy, amp in hotspots:
        b += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
    return b


def steady_state(b: np.ndarray, c: float, tol: float = 1e-12) -> np.ndarray:
    """Solve A x = b with matrix-free conjugate gradients: the answer the loop wants."""
    x = np.zeros_like(b)
    r = b - apply_A(x, c)
    p = r.copy()
    rr = float((r * r).sum())
    for _ in range(20 * b.size):
        Ap = apply_A(p, c)
        alpha = rr / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        rr_new = float((r * r).sum())
        if np.sqrt(rr_new) <= tol * np.linalg.norm(b):
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def time_constant(m: int, c: float) -> float:
    """tau = 1 / lambda_min(A): the die's thermal time constant, in the same units as dt.

    Dirichlet 5-point stencil eigenvalues are c + 4 - 2cos(p pi/(m+1)) - 2cos(q pi/(m+1)).
    """
    return 1.0 / (c + 4.0 - 4.0 * np.cos(np.pi / (m + 1)))


# ----------------------------------------------------------------- the transient
def relax(b: np.ndarray, c: float, u0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """`steps` explicit-Euler steps of du/dt = b - A u  (this is Richardson iteration)."""
    u = u0
    for _ in range(steps):
        u = u + dt * (b - apply_A(u, c))
    return u


def build(args):
    m, c = args.m, args.screening

    b_early = power_map(m, [(0.30, 0.32, 1.0), (0.68, 0.66, 0.75)], args.width)
    b_late = power_map(m, [(0.30, 0.32, 0.15), (0.68, 0.66, 0.9), (0.80, 0.22, 1.0)], args.width)

    x_early, x_late = steady_state(b_early, c), steady_state(b_late, c)

    dt = args.cfl / (4.0 + c)                    # explicit Euler is stable for dt < 2/lambda_max
    tau = time_constant(m, c)
    steps_per_frame = max(1, int(round(args.tau_per_frame * tau / dt)))

    switch_frame = args.frames // 3 if args.switch else None

    # --- roll the whole trajectory first: cheap here, and it keeps drawing simple
    u = np.zeros((m, m))
    fields, dists, targets = [], [], []
    for k in range(args.frames):
        b = b_late if (switch_frame is not None and k >= switch_frame) else b_early
        x = x_late if (switch_frame is not None and k >= switch_frame) else x_early
        if k:
            u = relax(b, c, u, dt, steps_per_frame)
        fields.append(u.copy())
        dists.append(np.linalg.norm(u - x) / np.linalg.norm(x))
        targets.append(x)

    t = np.arange(args.frames) * steps_per_frame * dt / tau     # time in units of tau
    vmax = max(x_early.max(), x_late.max())
    return dict(fields=fields, dists=np.array(dists), targets=targets, t=t,
                vmax=vmax, switch_frame=switch_frame, tau=tau, dt=dt,
                steps_per_frame=steps_per_frame)


# ----------------------------------------------------------------- writing it out
def even_pixels(fig, dpi: float) -> None:
    """Nudge the figure up to an even pixel count in both directions.

    H.264 with 4:2:0 chroma needs even width and height, and matplotlib sizes the
    canvas by *truncating* inches * dpi: the die-only panel at 4.6 x 3.9 in, 110 dpi
    comes out 505 x 430, which ffmpeg refuses outright. Ask the canvas what it
    actually is rather than recomputing it, then round up with half a pixel of slack
    so the float doesn't truncate straight back down.
    """
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
        elif args.out:                       # --out with --format both: keep the stem
            out = f"{args.out.rsplit('.', 1)[0]}.{fmt}"
        else:
            out = f"{stem}.{fmt}"
        if fmt == "gif":
            writer = PillowWriter(fps=args.fps)
        else:
            even_pixels(fig, args.dpi)
            # yuv420p + libx264 is the combination Keynote, PowerPoint and browsers
            # will actually play; matplotlib's default is yuv444p, which they won't.
            writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=args.bitrate,
                                  extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high",
                                              "-preset", "slow", "-movflags", "+faststart"])
        anim.save(out, writer=writer, dpi=args.dpi)
        written.append(out)
    return written


def animate(sim, args):
    m, c = args.m, args.screening
    t, dists = sim["t"], sim["dists"]
    die_only = args.panels == "die"

    if die_only:
        fig, ax_die = plt.subplots(figsize=(4.6, 3.9))
        ax_err = None
    else:
        fig, (ax_die, ax_err) = plt.subplots(
            1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    fig.patch.set_facecolor("white")

    im = ax_die.imshow(sim["fields"][0], cmap="inferno", vmin=0.0, vmax=sim["vmax"],
                       interpolation="nearest", origin="upper")
    for k in range(m + 1):
        ax_die.axhline(k - 0.5, color="white", lw=0.35, alpha=0.35)
        ax_die.axvline(k - 0.5, color="white", lw=0.35, alpha=0.35)
    ax_die.set_xticks([]); ax_die.set_yticks([])
    die_title = ax_die.set_title("", fontsize=11, color="#222")
    cb = fig.colorbar(im, ax=ax_die, fraction=0.046, pad=0.03)
    cb.set_label("temperature above ambient", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    if die_only:
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)          # keep the time read-out clear of the frame edge

        def draw_die(k):
            im.set_data(sim["fields"][k])
            # die_title.set_text(f"the die relaxing:  $t/\\tau$ = {t[k]:5.2f}")
            return im, die_title

        anim = FuncAnimation(fig, draw_die, frames=len(t), interval=1000 / args.fps, blit=False)
        written = save(anim, fig, args, "heat-dissipation-die")
        plt.close(fig)
        return written

    ax_err.semilogy(t, dists, color=PAL["gray"], lw=1.0, alpha=0.45)
    (trace,) = ax_err.plot([], [], color=PAL["red"], lw=2.4)
    (head,) = ax_err.plot([], [], "o", color=PAL["red"], ms=6)
    ax_err.set_xlim(t[0], t[-1])
    lo = max(dists.min() * 0.5, 1e-6)
    ax_err.set_ylim(lo, max(2.0, dists.max() * 1.4))
    ax_err.set_xlabel(r"time  $t/\tau$")
    ax_err.set_ylabel(r"$\|u(t)-A^{-1}b\|\,/\,\|A^{-1}b\|$")
    # ax_err.set_title("distance from the answer the control loop wants", fontsize=11)
    ax_err.grid(alpha=0.25)

    if args.deadline is not None:
        ax_err.axvline(args.deadline, color=PAL["navy"], ls="--", lw=1.2)
        ax_err.text(args.deadline, ax_err.get_ylim()[1], "  control period", color=PAL["navy"],
                    fontsize=9, va="top", ha="left")
    if sim["switch_frame"] is not None:
        ax_err.axvline(t[sim["switch_frame"]], color=PAL["orange"], ls=":", lw=1.6)
        ax_err.text(t[sim["switch_frame"]], lo * 1.6, " workload moves", color=PAL["orange"],
                    fontsize=9, ha="left")

    fig.tight_layout()

    def draw(k):
        im.set_data(sim["fields"][k])
        trace.set_data(t[: k + 1], dists[: k + 1])
        head.set_data([t[k]], [dists[k]])
        die_title.set_text(f"the die relaxing:  $t/\\tau$ = {t[k]:5.2f}     "
                           f"(screening $c$ = {c:g})")
        return im, trace, head, die_title

    anim = FuncAnimation(fig, draw, frames=len(t), interval=1000 / args.fps, blit=False)

    written = save(anim, fig, args, "heat-dissipation")
    plt.close(fig)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=24, help="tiles per side (default: 24, as in §4)")
    p.add_argument("--screening", type=float, default=0.4,
                   help="c: coupling to the coolant (default: 0.4; try 0.0 for a badly cooled die)")
    p.add_argument("--width", type=float, default=0.10, help="workload footprint (default: 0.10)")
    p.add_argument("--frames", type=int, default=120, help="frames in the animation")
    p.add_argument("--tau-per-frame", type=float, default=0.035,
                   help="how much of a thermal time constant passes per frame")
    p.add_argument("--cfl", type=float, default=0.45,
                   help="explicit-Euler step as a fraction of the stability limit (<0.5)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--switch", action="store_true",
                   help="move the workload a third of the way in: the target moves while you solve")
    p.add_argument("--deadline", type=float, default=None,
                   help="draw a control period at this t/tau")
    p.add_argument("--panels", choices=("both", "die"), default="both",
                   help="'both': die + distance-to-steady-state; 'die': the die alone, "
                        "sized to drop into a slide next to text")
    p.add_argument("--format", choices=("gif", "mp4", "both"), default="gif",
                   help="'mp4' needs ffmpeg on the PATH; 'both' writes the gif and the mp4")
    p.add_argument("--bitrate", type=int, default=2400, help="mp4 bitrate in kbit/s")
    p.add_argument("--out", default=None, help="output file (default: heat-dissipation.<format>)")
    args = p.parse_args()

    sim = build(args)
    written = animate(sim, args)
    print(f"{', '.join(written)}   {args.m}x{args.m} die, c = {args.screening}, "
          f"tau = {sim['tau']:.2f}, dt = {sim['dt']:.3f}, "
          f"{sim['steps_per_frame']} steps/frame, {args.frames} frames")
    print(f"reaches {sim['dists'][-1]:.2e} of the steady state by t/tau = {sim['t'][-1]:.2f}")


if __name__ == "__main__":
    main()
