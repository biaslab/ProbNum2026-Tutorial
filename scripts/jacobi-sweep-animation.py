# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""Jacobi, one component at a time, on the two-dimensional tiling of the die.

ProbNum 2026 tutorial, §2. Companion to `heat-dissipation-animation.py`: that one
shows the die relaxing as a physical transient, this one shows the *solver* doing
it, tile by tile, next to the matrix row it is reading.

The point of drawing it on the grid rather than the chain is the third panel. On a
single row of tiles $A$ is tridiagonal and row $i$ reaches $i\\pm1$ — the neighbour
is the previous unknown. On the tiling, row $i$ has *five* non-zeros: the diagonal,
the $\\pm 1$ band (torn wherever a row of tiles ends), and a $\\pm m$ band sitting
$m$ columns out. "The tile above me" is not "the unknown before me". The update

    x_i  <-  ( b_i + sum_{j ~ i} x_j ) / (4 + c)

is unchanged and still costs five numbers, but the locality is a property of the
*grid*, not of the matrix bandwidth. That gap is the whole reason the algorithm
wants to be phrased on the graph.

Two things the animation is built to make visible:

* **Jacobi is synchronous.** The left panel is $x^{(t)}$ and it does not move while
  the sweep runs; the right panel is $x^{(t+1)}$, filling in tile by tile. Every
  tile reads the *old* field. Because of that the visiting order is irrelevant —
  `--order redblack` or `--order random` produce a bit-identical field at each
  commit, which is worth showing rather than asserting. Pass `--gauss-seidel` to
  break exactly that: the read panel then changes under the sweep and the order
  starts to matter.
* **The centre tile's own old value is never read.** It is outlined dotted, not
  solid: $x_i$ does not appear on the right-hand side of its own equation. That is
  the deletion that turns row $i$ into a message.

Usage
-----
    uv run python/jacobi-sweep-animation.py                     # 6x6 die -> figures/jacobi-sweep.gif
    uv run python/jacobi-sweep-animation.py --order redblack    # same commits, different visiting order
    uv run python/jacobi-sweep-animation.py --gauss-seidel      # in place: the read panel moves too
    uv run python/jacobi-sweep-animation.py --panels grids      # drop the matrix, narrow slide
    uv run python/jacobi-sweep-animation.py --format mp4        # needs ffmpeg

Without uv:  pip install numpy matplotlib pillow  &&  python python/jacobi-sweep-animation.py
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

PAL = dict(blue="#2a78d6", red="#C81919", navy="#101073", gray="#898781",
           orange="#eb6834", white="#fcfcfb", green="#008300")


# ----------------------------------------------------------------- the operator
def grid_matrix(m: int, c: float) -> np.ndarray:
    """A = (c - Delta) on an m x m tiling, unknowns numbered row by row: k = i*m + j.

    Dense on purpose: m is small enough to draw, and the animation wants to read
    whole rows of A. Ambient (zero) outside the die, so the diagonal is 4 + c
    everywhere -- an edge tile conducts its missing neighbours' share to coolant.
    """
    n = m * m
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            A[k, k] = 4.0 + c
            if j + 1 < m:
                A[k, k + 1] = A[k + 1, k] = -1.0        # the tile beside it: +-1
            if i + 1 < m:
                A[k, k + m] = A[k + m, k] = -1.0        # the tile below it:  +-m
    return A


def power_map(m: int, hotspots, width: float) -> np.ndarray:
    """Dissipated power b: a few Gaussian workloads on the die, in (m, m) form."""
    g = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(g, g, indexing="ij")
    b = np.zeros((m, m))
    for cx, cy, amp in hotspots:
        b += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
    return b


def neighbours(i: int, j: int, m: int):
    """The four tiles tile (i, j) touches, clipped to the die."""
    return [(a, c) for a, c in ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
            if 0 <= a < m and 0 <= c < m]


def visiting_order(n: int, m: int, kind: str, seed: int) -> np.ndarray:
    """Which component to update when. For Jacobi this cannot change the result."""
    if kind == "lex":
        return np.arange(n)
    if kind == "redblack":
        k = np.arange(n)
        red = (k // m + k % m) % 2 == 0
        return np.concatenate([k[red], k[~red]])
    return np.random.default_rng(seed).permutation(n)


# ----------------------------------------------------------------- the trajectory
def build(args):
    """Roll the whole sweep sequence up front; one frame per component update."""
    m, c = args.m, args.screening
    n = m * m
    A = grid_matrix(m, c)
    b2 = power_map(m, [(0.30, 0.32, 1.0), (0.70, 0.66, 0.8)], args.width)
    b = b2.ravel()
    bn = np.linalg.norm(b)
    x_star = np.linalg.solve(A, b)

    order = visiting_order(n, m, args.order, args.seed)
    residual = lambda v: float(np.linalg.norm(b - A @ v) / bn)

    x = np.zeros(n)
    for _ in range(args.warmup):
        # Silent sweeps before the camera rolls. From x = 0 the first sweep is just
        # x = D^{-1}b -- no coupling is exercised, and the read panel is blank. One
        # warm-up sweep gives the animated sweeps a field with structure to read.
        x = x + (b - A @ x) / np.diag(A)
    res_now = residual(x)
    frames, commits = [], []

    for t in range(args.sweeps):
        x_read = x.copy()                     # what the whole sweep is allowed to read
        pending = np.full(n, np.nan)          # x^(t+1), NaN where not yet computed
        for step, k in enumerate(order):
            src = x if args.gauss_seidel else x_read
            row = A[k]
            off = float(row @ src - row[k] * src[k])      # the coupling, x_k excluded
            val = (b[k] - off) / row[k]
            if args.gauss_seidel:
                x[k] = val
            pending[k] = val
            i, j = divmod(k, m)
            nb = neighbours(i, j, m)
            frames.append(dict(
                phase="update", sweep=t, step=step, k=k, i=i, j=j, nb=nb,
                read=(x if args.gauss_seidel else x_read).copy(),
                write=pending.copy(), val=val,
                nbsum=float(sum(src[a * m + cc] for a, cc in nb)),
                bi=float(b[k]), diag=float(row[k]), res=res_now))
        x = np.where(np.isnan(pending), x, pending)
        res_now = residual(x)
        commits.append(res_now)
        for _ in range(args.hold):
            frames.append(dict(
                phase="hold", sweep=t, step=len(order), k=-1, i=-1, j=-1, nb=[],
                read=x_read.copy(), write=pending.copy(), val=np.nan,
                nbsum=np.nan, bi=np.nan, diag=float(A[0, 0]), res=res_now))

    err = float(np.linalg.norm(x - x_star) / np.linalg.norm(x_star))
    return dict(frames=frames, A=A, b=b, m=m, n=n, order=order, commits=commits,
                vmax=float(x_star.max()), err=err, x_star=x_star)


# ----------------------------------------------------------------- drawing
def die_panel(ax, m: int, field, vmax: float, cmap):
    """One tiled die, colour-mapped, with the tile boundaries drawn in."""
    im = ax.imshow(field, cmap=cmap, vmin=0.0, vmax=vmax,
                   interpolation="nearest", origin="upper")
    for k in range(m + 1):
        ax.axhline(k - 0.5, color="white", lw=0.5, alpha=0.5)
        ax.axvline(k - 0.5, color="white", lw=0.5, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def matrix_panel(ax, A, m: int):
    """The sparsity of A: white where zero, blue off-diagonal, dark blue diagonal."""
    n = A.shape[0]
    code = np.zeros((n, n))
    code[A != 0] = 1.0
    np.fill_diagonal(code, 2.0)
    cmap = ListedColormap(["white", "#a9caf0", PAL["blue"]])
    ax.imshow(code, cmap=cmap, vmin=-0.5, vmax=2.5,
              interpolation="nearest", origin="upper")
    step = 1 if n <= 40 else m
    for k in range(0, n + 1, step):
        ax.axhline(k - 0.5, color="#dcdce2", lw=0.35)
        ax.axvline(k - 0.5, color="#dcdce2", lw=0.35)
    for k in range(0, n + 1, m):                     # block boundaries: rows of tiles
        ax.axhline(k - 0.5, color=PAL["gray"], lw=0.7, alpha=0.6)
        ax.axvline(k - 0.5, color=PAL["gray"], lw=0.7, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    return cmap


def cell(x, y, inset: float = 0.0):
    """Lower-left corner of the imshow cell centred at column x, row y."""
    return (x - 0.5 + inset, y - 0.5 + inset)


def animate(sim, args):
    m, n = sim["m"], sim["n"]
    frames, A = sim["frames"], sim["A"]
    method = "Gauss–Seidel" if args.gauss_seidel else "Jacobi"

    cmap = copy.copy(plt.cm.inferno)
    cmap.set_bad("#e9e9ee")                            # tiles not yet written

    show_read = args.panels in ("all", "grids")
    show_mat = args.panels in ("all", "matrix")
    widths = ([1.0] if show_read else []) + [1.0] + ([1.35] if show_mat else [])
    figw = 4.5 * len(widths) + (0.6 if show_mat else 0.0)
    fig, axes = plt.subplots(1, len(widths), figsize=(figw, 4.5),
                             gridspec_kw=dict(width_ratios=widths))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    a = 0
    ax_read = axes[a] if show_read else None
    a += int(show_read)
    ax_write = axes[a]
    a += 1
    ax_mat = axes[a] if show_mat else None

    f0 = frames[0]
    write0 = np.ma.masked_invalid(f0["write"].reshape(m, m))

    if show_read:
        im_read = die_panel(ax_read, m, f0["read"].reshape(m, m), sim["vmax"], cmap)
        ax_read.set_title(
            r"read:  $x^{(t)}$" if not args.gauss_seidel else r"read:  $x$, in place",
            fontsize=11.5, color="#222")
        # the four tiles row i reaches, and the centre it does *not* read
        nb_rects = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["blue"],
                              lw=2.4, visible=False) for _ in range(4)]
        self_rect = Rectangle((0, 0), 0.72, 0.72, facecolor="none", edgecolor=PAL["red"],
                              lw=1.8, ls=(0, (2.5, 1.5)), visible=False)
        for r in nb_rects + [self_rect]:
            ax_read.add_patch(r)
        ax_read.legend(
            handles=[Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["blue"],
                               lw=2.4, label="the four tiles it reads"),
                     Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["red"],
                               lw=1.8, ls=(0, (2.5, 1.5)),
                               label="its own value (never read)")],
            loc="upper center", bbox_to_anchor=(0.5, -0.015), frameon=False,
            fontsize=9.5, handlelength=1.4, borderpad=0.2, labelspacing=0.35)

    im_write = die_panel(ax_write, m, write0, sim["vmax"], cmap)
    ax_write.set_title(r"write:  $x^{(t+1)}$" if not args.gauss_seidel
                       else r"write:  this sweep's values", fontsize=11.5, color="#222")
    tgt_rect = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["red"],
                         lw=2.8, visible=False)
    ax_write.add_patch(tgt_rect)
    ax_write.set_xlabel("red: the tile being written\ngrey: not computed yet",
                        fontsize=9.5, color="#555", labelpad=8)
    cb = fig.colorbar(im_write, ax=ax_write, fraction=0.046, pad=0.03)
    cb.set_label("temperature above ambient", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    if show_mat:
        matrix_panel(ax_mat, A, m)
        ax_mat.set_title(r"row $i$ of $A$",
                         fontsize=11.5, color="#222")
        # ax_mat.set_xlabel("the $\\pm m$ band is the tile above and below —\n"
                        #   "on the tiling $A$ is not tridiagonal",
                        #   fontsize=9.5, color=PAL["gray"], labelpad=8)
        band = Rectangle((-0.5, 0), n, 1, facecolor=PAL["orange"], alpha=0.28,
                         edgecolor="none", visible=False)
        ax_mat.add_patch(band)
        # A single cell of a 64x64 matrix is a few pixels wide, so the five entries
        # get ring markers several cells across rather than cell-sized outlines.
        ring = max(1.0, n / 18.0)
        mat_rects = [Rectangle((0, 0), ring, ring, facecolor="none", edgecolor=PAL["blue"],
                               lw=1.8, visible=False) for _ in range(4)]
        mat_diag = Rectangle((0, 0), ring, ring, facecolor="none", edgecolor=PAL["red"],
                             lw=2.2, visible=False)
        for r in mat_rects + [mat_diag]:
            ax_mat.add_patch(r)

    head = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=12.5, color="#222")
    # calc = fig.text(0.5, 0.04, "", ha="center", va="center", fontsize=12.0,
    #                 color="#222", family="DejaVu Sans")

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.935))

    def draw(fi):
        f = frames[fi]
        im_write.set_data(np.ma.masked_invalid(f["write"].reshape(m, m)))
        if show_read:
            im_read.set_data(f["read"].reshape(m, m))

        if f["phase"] == "update":
            i, j, k = f["i"], f["j"], f["k"]
            tgt_rect.set_xy(cell(j, i)); tgt_rect.set_visible(True)
            if show_read:
                self_rect.set_xy(cell(j, i, 0.14)); self_rect.set_visible(True)
                for r, nb in zip(nb_rects, f["nb"] + [None] * 4):
                    if nb is None:
                        r.set_visible(False)
                    else:
                        r.set_xy(cell(nb[1], nb[0])); r.set_visible(True)
            if show_mat:
                band.set_xy((-0.5, k - 0.5)); band.set_visible(True)
                mat_diag.set_xy(cell(k, k, 0.5 - ring / 2)); mat_diag.set_visible(True)
                cols = [a * m + c for a, c in f["nb"]]
                for r, col in zip(mat_rects, cols + [None] * 4):
                    if col is None:
                        r.set_visible(False)
                    else:
                        r.set_xy(cell(col, k, 0.5 - ring / 2)); r.set_visible(True)

            head.set_text(f"{method} sweep {args.warmup + f['sweep'] + 1}"
                          f"   —   component {f['step'] + 1} of {n}:  tile $({i},{j})$,"
                          f" row $i = {k}$")
            # edge = "" if len(f["nb"]) == 4 else \
            #     f"      (edge tile: {len(f['nb'])} neighbours, the rest is coolant)"
            # calc.set_text(
            #     f"$x_{{{i},{j}}} \\leftarrow (b_i + \\sum_{{j \\sim i}} x_j)\\,/\\,(4+c)$"
            #     f"  $= ({f['bi']:.3f} + {f['nbsum']:.3f})\\,/\\,{f['diag']:.2f}"
            #     f" = {f['val']:.4f}$" + edge)
        else:
            tgt_rect.set_visible(False)
            if show_read:
                self_rect.set_visible(False)
                for r in nb_rects:
                    r.set_visible(False)
            if show_mat:
                band.set_visible(False); mat_diag.set_visible(False)
                for r in mat_rects:
                    r.set_visible(False)
            head.set_text(f"{method} sweep {args.warmup + f['sweep'] + 1} complete"
                          f"   —   all {n} components updated from the same $x^{{(t)}}$"
                          if not args.gauss_seidel else
                          f"{method} sweep {args.warmup + f['sweep'] + 1} complete"
                          f"   —   all {n} components updated, each from the latest values")
            # calc.set_text(f"$\\Vert b - Ax^{{(t+1)}}\\Vert/\\Vert b\\Vert = {f['res']:.3e}$"
            #               f"      (visiting order: {args.order}"
            #               + ("" if args.gauss_seidel else
            #                  " — for Jacobi the order cannot change this number)")
            #               + (")" if args.gauss_seidel else ""))
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    written = save(anim, fig, args, "jacobi-sweep" if not args.gauss_seidel
                   else "gauss-seidel-sweep")
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
        elif args.out:                       # --out with --format both: keep the stem
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
    p.add_argument("--m", type=int, default=8,
                   help="tiles per side (default: 8 — one sweep is m^2 frames)")
    p.add_argument("--screening", type=float, default=0.4,
                   help="c: coupling to the coolant (default: 0.4)")
    p.add_argument("--width", type=float, default=0.18, help="workload footprint")
    p.add_argument("--sweeps", type=int, default=2, help="Jacobi sweeps to animate")
    p.add_argument("--warmup", type=int, default=1,
                   help="sweeps run before the camera rolls, so the read panel has "
                        "a field with structure in it (default: 1)")
    p.add_argument("--hold", type=int, default=8,
                   help="frames to hold on the completed sweep")
    p.add_argument("--order", choices=("lex", "redblack", "random"), default="lex",
                   help="visiting order within a sweep (immaterial for Jacobi)")
    p.add_argument("--seed", type=int, default=0, help="seed for --order random")
    p.add_argument("--gauss-seidel", action="store_true",
                   help="update in place: the read panel moves under the sweep")
    p.add_argument("--panels", choices=("all", "grids", "matrix"), default="all",
                   help="'all': read + write + matrix; 'grids': the two dies; "
                        "'matrix': write die + matrix")
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
    nnz = int((sim["A"] != 0).sum())
    print(f"{', '.join(written)}   {args.m}x{args.m} die, n = {sim['n']} unknowns, "
          f"{nnz} non-zeros ({nnz / sim['n']:.2f} per row), c = {args.screening}, "
          f"order = {args.order}, {len(sim['frames'])} frames")
    print("relative residual after each sweep: "
          + ", ".join(f"{r:.3e}" for r in sim["commits"]))
    print(f"relative error against the exact solve: {sim['err']:.3e}")


if __name__ == "__main__":
    main()
