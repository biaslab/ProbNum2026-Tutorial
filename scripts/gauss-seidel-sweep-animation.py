# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///
"""Gauss–Seidel, one component at a time, on the two-dimensional tiling of the die.

ProbNum 2026 tutorial, §2. Sibling of `jacobi-sweep-animation.py`: same die, same
five-point stencil, same per-component pacing, so the two can be played back to
back. Everything that differs between the two methods is what this figure is
built to show, and it is one word — *in place*.

Jacobi keeps two fields: it reads $x^{(t)}$ and writes $x^{(t+1)}$, so the field it
reads is frozen for the whole sweep. Gauss–Seidel keeps one. When tile $i$ comes
up, whichever of its neighbours have already been visited this sweep hand it their
*new* values:

    x_i  <-  ( b_i + sum_{j<i} x_j^{new} + sum_{j>i} x_j^{old} ) / (4 + c)

Three consequences, one per panel:

* **The sweep front is real.** The live field carries a boundary between tiles that
  have been refreshed and tiles still holding last sweep's value (drawn hatched).
  Each update reads across it, and how many fresh values it gets depends on where
  in the sweep it sits — the head-of-frame counter says which.
* **It is strictly better informed than Jacobi**, and the second panel is the
  evidence: it holds $x^{(t)}$, the field Jacobi *would* have read, with the
  already-refreshed neighbours ringed. The residual after each sweep is printed
  against Jacobi's from the same starting field.
* **The order now matters.** With lexicographic order the refreshed columns are
  exactly the strict lower triangle, so the update is $(D+L)x^{(t+1)} = b - Ux^{(t)}$.
  Change `--order` and the green columns stop being a triangle — the splitting is
  $L$ and $U$ of the *permuted* matrix, and the residual it reaches is a different
  number. The caption reports both, because unlike Jacobi this is not invariant.

Read the caption's two comparison numbers with one caveat: both continue from the
*animated* run's warm-up field, so the first compared sweep is mildly flattered by
whatever order was warmed up. The gap is not an artefact of that — red-black really
does have a worse per-sweep factor here than lexicographic (0.29 against 0.21 after
four sweeps on the default problem) and buys a parallelisable sweep in exchange.
Reverse lexicographic, by contrast, is *not* a meaningful contrast on a symmetric
stencil: it is the transposed splitting and lands within half a percent.

Usage
-----
    uv run python/gauss-seidel-sweep-animation.py                  # -> figures/gauss-seidel-sweep.gif
    uv run python/gauss-seidel-sweep-animation.py --order redblack # the front is a checkerboard
    uv run python/gauss-seidel-sweep-animation.py --order reverse  # sweep the other way
    uv run python/gauss-seidel-sweep-animation.py --panels matrix  # die + matrix, no Jacobi panel
    uv run python/gauss-seidel-sweep-animation.py --format mp4     # needs ffmpeg

Without uv:  pip install numpy matplotlib pillow  &&  python python/gauss-seidel-sweep-animation.py
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
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Patch, Rectangle

PAL = dict(blue="#2a78d6", red="#C81919", navy="#101073", gray="#898781",
           orange="#eb6834", white="#fcfcfb", green="#008300")


# ----------------------------------------------------------------- the operator
def grid_matrix(m: int, c: float) -> np.ndarray:
    """A = (c - Delta) on an m x m tiling, unknowns numbered row by row: k = i*m + j.

    Same operator as the Jacobi animation and as `grid_matrix` in notebook 01;
    dense because m is small enough to draw and the animation reads whole rows.
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
    """Which component to update when. For Gauss-Seidel this *does* change the result."""
    k = np.arange(n)
    if kind == "lex":
        return k
    if kind == "reverse":
        return k[::-1].copy()
    if kind == "redblack":
        red = (k // m + k % m) % 2 == 0
        return np.concatenate([k[red], k[~red]])
    return np.random.default_rng(seed).permutation(n)


# ----------------------------------------------------------------- the two sweeps
def gs_sweep(A, b, x, order):
    """One in-place Gauss-Seidel sweep in the given order. Mutates and returns x."""
    for k in order:
        row = A[k]
        x[k] = (b[k] - (row @ x - row[k] * x[k])) / row[k]
    return x


def jacobi_sweep(A, b, x):
    """One Jacobi sweep: the whole field from the old field, order-independent."""
    return x + (b - A @ x) / np.diag(A)


def build(args):
    """Roll the whole sweep sequence up front; one frame per component update."""
    m, c = args.m, args.screening
    n = m * m
    A = grid_matrix(m, c)
    b = power_map(m, [(0.30, 0.32, 1.0), (0.70, 0.66, 0.8)], args.width).ravel()
    bn = np.linalg.norm(b)
    x_star = np.linalg.solve(A, b)
    diag = np.diag(A)

    order = visiting_order(n, m, args.order, args.seed)
    alt_name = args.compare_order
    if alt_name == "auto":
        # Reverse lexicographic is a poor contrast: on a symmetric stencil it is the
        # transposed splitting and lands within half a percent. Red-black is the
        # honest one -- a genuinely different convergence factor, and the ordering
        # people actually use when they want the sweep to parallelise.
        alt_name = "redblack" if args.order != "redblack" else "lex"
    alt_order = visiting_order(n, m, alt_name, args.seed)
    residual = lambda v: float(np.linalg.norm(b - A @ v) / bn)

    x = np.zeros(n)
    for _ in range(args.warmup):
        # Silent sweeps before the camera rolls. From x = 0 the first sweep reads a
        # blank field, so there is no "fresh vs stale" to see; one warm-up sweep
        # gives the animated ones a field with structure in it.
        x = gs_sweep(A, b, x, order)

    # Same starting field, three other ways of continuing: Jacobi, and Gauss-Seidel
    # in a different order. These are the numbers the caption compares against.
    x_jac, x_alt = x.copy(), x.copy()
    jac_commits, alt_commits = [], []

    res_now = residual(x)
    frames, commits = [], []

    for t in range(args.sweeps):
        x_read = x.copy()                     # what Jacobi would have read all sweep
        visited = np.zeros(n, dtype=bool)
        for step, k in enumerate(order):
            row = A[k]
            off = float(row @ x - row[k] * x[k])         # the coupling, x_k excluded
            val = (b[k] - off) / row[k]
            i, j = divmod(k, m)
            nb = neighbours(i, j, m)
            fresh = [(a, cc) for a, cc in nb if visited[a * m + cc]]
            stale = [(a, cc) for a, cc in nb if not visited[a * m + cc]]
            x[k] = val
            visited[k] = True
            frames.append(dict(
                phase="update", sweep=t, step=step, k=k, i=i, j=j,
                nb_fresh=fresh, nb_stale=stale, nb_count=len(nb),
                live=x.copy(), frozen=x_read.copy(), visited=visited.copy(),
                val=val, bi=float(b[k]), diag=float(row[k]),
                sum_fresh=float(sum(x[a * m + cc] for a, cc in fresh)),
                sum_stale=float(sum(x[a * m + cc] for a, cc in stale)),
                res=res_now, jac=np.nan, alt=np.nan))
        res_now = residual(x)
        commits.append(res_now)
        x_jac = jacobi_sweep(A, b, x_jac)
        jac_commits.append(residual(x_jac))
        x_alt = gs_sweep(A, b, x_alt, alt_order)
        alt_commits.append(residual(x_alt))
        for _ in range(args.hold):
            frames.append(dict(
                phase="hold", sweep=t, step=len(order), k=-1, i=-1, j=-1,
                nb_fresh=[], nb_stale=[], nb_count=0,
                live=x.copy(), frozen=x_read.copy(), visited=visited.copy(),
                val=np.nan, bi=np.nan, diag=float(diag[0]),
                sum_fresh=np.nan, sum_stale=np.nan,
                res=res_now, jac=jac_commits[-1], alt=alt_commits[-1]))

    err = float(np.linalg.norm(x - x_star) / np.linalg.norm(x_star))
    return dict(frames=frames, A=A, b=b, m=m, n=n, order=order, commits=commits,
                jac_commits=jac_commits, alt_commits=alt_commits, alt_name=alt_name,
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
              interpolation="nearest", origin="upper", zorder=1)
    step = 1 if n <= 40 else m
    for k in range(0, n + 1, step):
        ax.axhline(k - 0.5, color="#dcdce2", lw=0.35, zorder=3)
        ax.axvline(k - 0.5, color="#dcdce2", lw=0.35, zorder=3)
    for k in range(0, n + 1, m):                     # block boundaries: rows of tiles
        ax.axhline(k - 0.5, color=PAL["gray"], lw=0.7, alpha=0.6, zorder=3)
        ax.axvline(k - 0.5, color=PAL["gray"], lw=0.7, alpha=0.6, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)


def cell(x, y, inset: float = 0.0):
    """Lower-left corner of the imshow cell centred at column x, row y."""
    return (x - 0.5 + inset, y - 0.5 + inset)


def animate(sim, args):
    m, n = sim["m"], sim["n"]
    frames, A = sim["frames"], sim["A"]

    cmap = copy.copy(plt.cm.inferno)

    show_jac = args.panels in ("all", "compare")
    show_mat = args.panels in ("all", "matrix")
    widths = [1.0] + ([1.0] if show_jac else []) + ([1.35] if show_mat else [])
    # With one panel there is no room for the read-outs on a single line: the
    # head-of-frame text breaks over three lines and takes the panel title's space.
    narrow = len(widths) == 1
    figw = 4.5 * len(widths) + (0.6 if show_mat else 0.0)
    fig, axes = plt.subplots(1, len(widths), figsize=(figw, 4.5),
                             gridspec_kw=dict(width_ratios=widths))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    ax_live = axes[0]
    ax_jac = axes[1] if show_jac else None
    ax_mat = axes[1 + int(show_jac)] if show_mat else None

    f0 = frames[0]

    # ---- panel 1: the one field, overwritten in place
    im_live = die_panel(ax_live, m, f0["live"].reshape(m, m), sim["vmax"], cmap)
    ax_live.set_title("" if narrow else "the field, overwritten in place",
                      fontsize=11.5, color="#222")
    # hatching marks tiles that still hold last sweep's value: the sweep front is
    # wherever the hatching stops, and every update reads across it.
    stale_hatch = [Rectangle(cell(j, i), 1, 1, facecolor="none", edgecolor="white",
                             hatch="////", lw=0.0, alpha=0.55, visible=False, zorder=2)
                   for i in range(m) for j in range(m)]
    for r in stale_hatch:
        ax_live.add_patch(r)
    fresh_rects = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["green"],
                             lw=2.6, visible=False, zorder=4) for _ in range(4)]
    stale_rects = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["blue"],
                             lw=2.6, visible=False, zorder=4) for _ in range(4)]
    tgt_rect = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PAL["red"],
                         lw=2.8, visible=False, zorder=5)
    for r in fresh_rects + stale_rects + [tgt_rect]:
        ax_live.add_patch(r)
    ax_live.legend(
        handles=[Patch(facecolor="none", edgecolor=PAL["red"], lw=2.8,
                       label="being written now"),
                 Patch(facecolor="none", edgecolor=PAL["green"], lw=2.6,
                       label="neighbour, already refreshed this sweep"),
                 Patch(facecolor="none", edgecolor=PAL["blue"], lw=2.6,
                       label="neighbour, still last sweep's value"),
                 Patch(facecolor="white", edgecolor=PAL["gray"], hatch="////", lw=0.5,
                       label="hatched: not yet visited this sweep")],
        loc="upper center", bbox_to_anchor=(0.5, -0.015), frameon=False,
        fontsize=8.5, handlelength=1.3, borderpad=0.2, labelspacing=0.3)

    cb_ax = ax_jac if show_jac else ax_live

    # ---- panel 2: the field Jacobi would have read instead
    if show_jac:
        im_jac = die_panel(ax_jac, m, f0["frozen"].reshape(m, m), sim["vmax"], cmap)
        ax_jac.set_title(r"$x^{(t)}$ — what Jacobi would have read",
                         fontsize=11.5, color="#222")
        ax_jac.set_xlabel("frozen for the whole sweep;\n"
                          "grey rings: the values Gauss–Seidel has already improved",
                          fontsize=9.0, color="#555", labelpad=8)
        jac_rects = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#d8d8dc",
                               lw=2.2, ls=(0, (2.5, 1.5)), visible=False, zorder=4)
                     for _ in range(4)]
        for r in jac_rects:
            ax_jac.add_patch(r)

    cb = fig.colorbar(im_live, ax=cb_ax, fraction=0.046, pad=0.03)
    cb.set_label("temperature above ambient", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- panel 3: the row of A, and which of its columns are already refreshed
    if show_mat:
        matrix_panel(ax_mat, A, m)
        ax_mat.set_title(r"row $i$ of $A$:  which columns are already refreshed",
                         fontsize=11.5, color="#222")
        # A tint over the columns whose unknown has been updated this sweep. In
        # lexicographic order that region *is* the strict lower triangle, i.e. the L
        # of (D+L)x = b - Ux; in any other order it is L of the permuted matrix,
        # which is the honest reason --order changes the answer.
        tint = np.zeros((n, n, 4))
        ov = ax_mat.imshow(tint, interpolation="nearest", origin="upper", zorder=2)
        green_rgba = np.array(to_rgba(PAL["green"], 0.16))
        band = Rectangle((-0.5, 0), n, 1, facecolor=PAL["orange"], alpha=0.28,
                         edgecolor="none", visible=False, zorder=4)
        ax_mat.add_patch(band)
        # A single cell of a 64x64 matrix is a few pixels wide, so the entries get
        # ring markers several cells across rather than cell-sized outlines.
        ring = max(1.0, n / 18.0)
        mat_rects = [Rectangle((0, 0), ring, ring, facecolor="none", lw=1.9,
                               edgecolor=PAL["blue"], visible=False, zorder=5)
                     for _ in range(4)]
        mat_diag = Rectangle((0, 0), ring, ring, facecolor="none", edgecolor=PAL["red"],
                             lw=2.2, visible=False, zorder=6)
        for r in mat_rects + [mat_diag]:
            ax_mat.add_patch(r)
        refreshed_label = ("columns already refreshed  ($L$: the strict lower triangle)"
                           if args.order == "lex" else
                           "columns already refreshed  (no longer a triangle:\n"
                           f"$L$ of the matrix permuted into {args.order} order)")
        ax_mat.legend(
            handles=[Patch(facecolor=to_rgba(PAL["green"], 0.30), edgecolor="none",
                           label=refreshed_label),
                     Patch(facecolor=PAL["orange"], alpha=0.4, edgecolor="none",
                           label="row $i$: the equation being solved")],
            loc="upper center", bbox_to_anchor=(0.5, -0.015), frameon=False,
            fontsize=8.5, handlelength=1.3, borderpad=0.2, labelspacing=0.3)

    # `sep` and `gap` are the separators that become newlines when they have to.
    sep = "\n" if narrow else "   —   "
    gap = "\n" if narrow else "      "
    head = fig.text(0.5, 0.975, "", ha="center", va="top",
                    fontsize=10.5 if narrow else 12.5, color="#222")
    calc = fig.text(0.5, 0.055 if narrow else 0.038, "", ha="center", va="center",
                    fontsize=10.0 if narrow else 11.5, color="#222",
                    family="DejaVu Sans")

    fig.tight_layout(rect=(0.0, 0.16 if narrow else 0.10, 1.0,
                           0.885 if narrow else 0.935))

    def draw(fi):
        f = frames[fi]
        im_live.set_data(f["live"].reshape(m, m))
        if show_jac:
            im_jac.set_data(f["frozen"].reshape(m, m))
        vis = f["visited"]
        for k, r in enumerate(stale_hatch):
            r.set_visible(not vis[k])
        if show_mat:
            tint[...] = 0.0
            tint[:, vis] = green_rgba
            ov.set_data(tint)

        if f["phase"] == "update":
            i, j, k = f["i"], f["j"], f["k"]
            tgt_rect.set_xy(cell(j, i)); tgt_rect.set_visible(True)
            for rects, nbs in ((fresh_rects, f["nb_fresh"]), (stale_rects, f["nb_stale"])):
                for r, nb in zip(rects, nbs + [None] * 4):
                    if nb is None:
                        r.set_visible(False)
                    else:
                        r.set_xy(cell(nb[1], nb[0])); r.set_visible(True)
            if show_jac:
                # the same cells, in the field Jacobi would have used: these are
                # exactly the entries where the two methods now disagree
                for r, nb in zip(jac_rects, f["nb_fresh"] + [None] * 4):
                    if nb is None:
                        r.set_visible(False)
                    else:
                        r.set_xy(cell(nb[1], nb[0])); r.set_visible(True)
            if show_mat:
                band.set_xy((-0.5, k - 0.5)); band.set_visible(True)
                mat_diag.set_xy(cell(k, k, 0.5 - ring / 2)); mat_diag.set_visible(True)
                cols = [(a * m + cc, True) for a, cc in f["nb_fresh"]] + \
                       [(a * m + cc, False) for a, cc in f["nb_stale"]]
                for r, item in zip(mat_rects, cols + [None] * 4):
                    if item is None:
                        r.set_visible(False)
                    else:
                        col, is_fresh = item
                        r.set_edgecolor(PAL["green"] if is_fresh else PAL["blue"])
                        r.set_xy(cell(col, k, 0.5 - ring / 2)); r.set_visible(True)

            head.set_text(
                f"Gauss–Seidel sweep {args.warmup + f['sweep'] + 1}"
                f"{sep}component {f['step'] + 1} of {n}:  tile $({i},{j})$,"
                f" row $i = {k}$"
                f"{sep}{len(f['nb_fresh'])} of {f['nb_count']} neighbours already refreshed")
            rule = (f"$x_{{{i},{j}}} \\leftarrow (b_i + \\sum_{{j<i}} x_j^{{new}}"
                    f" + \\sum_{{j>i}} x_j^{{old}})\\,/\\,(4+c)$")
            numbers = (f"$= ({f['bi']:.3f} + {f['sum_fresh']:.3f}_{{\\ new}}"
                       f" + {f['sum_stale']:.3f}_{{\\ old}})\\,/\\,{f['diag']:.2f}"
                       f" = {f['val']:.4f}$")
            calc.set_text(rule + ("\n" if narrow else "   ") + numbers)
        else:
            tgt_rect.set_visible(False)
            for r in fresh_rects + stale_rects:
                r.set_visible(False)
            if show_jac:
                for r in jac_rects:
                    r.set_visible(False)
            if show_mat:
                band.set_visible(False); mat_diag.set_visible(False)
                for r in mat_rects:
                    r.set_visible(False)
            head.set_text(f"Gauss–Seidel sweep {args.warmup + f['sweep'] + 1} complete"
                          f"{sep}every tile now holds a value from this sweep")
            calc.set_text(
                f"$\\Vert b - Ax\\Vert/\\Vert b\\Vert = {f['res']:.3e}$"
                f"{gap}Jacobi from the same field: ${f['jac']:.3e}$"
                f"{gap}Gauss–Seidel in {sim['alt_name']} order: ${f['alt']:.3e}$")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    written = save(anim, fig, args, "gauss-seidel-sweep")
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
    p.add_argument("--sweeps", type=int, default=2, help="sweeps to animate")
    p.add_argument("--warmup", type=int, default=1,
                   help="sweeps run before the camera rolls, so the field being read "
                        "has structure in it (default: 1)")
    p.add_argument("--hold", type=int, default=8,
                   help="frames to hold on the completed sweep")
    p.add_argument("--order", choices=("lex", "reverse", "redblack", "random"),
                   default="lex",
                   help="visiting order within a sweep — for Gauss-Seidel this changes "
                        "the iterate, and only 'lex' makes the refreshed set the "
                        "lower triangle of A")
    p.add_argument("--seed", type=int, default=0, help="seed for --order random")
    p.add_argument("--compare-order", choices=("auto", "lex", "reverse", "redblack",
                                              "random"), default="auto",
                   help="the order the caption's second Gauss-Seidel run uses, "
                        "continuing from the same field (default: red-black, or "
                        "lexicographic if that is what is being animated)")
    p.add_argument("--panels", choices=("all", "die", "matrix", "compare"), default="all",
                   help="'all': live die + Jacobi's frozen field + matrix; "
                        "'die': the live die alone; 'matrix': live die + matrix; "
                        "'compare': live die + Jacobi's frozen field")
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
    rows = (("Gauss-Seidel, " + args.order, sim["commits"]),
            ("Gauss-Seidel, " + sim["alt_name"], sim["alt_commits"]),
            ("Jacobi", sim["jac_commits"]))
    for name, cs in rows:
        print(f"  relative residual per sweep, {name:<26} "
              + ", ".join(f"{r:.3e}" for r in cs))
    print(f"relative error against the exact solve: {sim['err']:.3e}")


if __name__ == "__main__":
    main()
