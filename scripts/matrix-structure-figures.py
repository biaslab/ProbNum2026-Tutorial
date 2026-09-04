# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""The two matrices behind the two topologies of §1 — drawn in the style of the graph figure.

Companion to `topologies.png`: same palette, same pairing. The chain's matrix is tridiagonal;
the die's matrix is the five-point stencil, whose extra bands sit m columns out and whose
first off-diagonal is *torn* at every row boundary of the die.

    uv run python/matrix-structure-figures.py            # writes both PNGs into figures/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

GREEN = "#008300"; BLUE = "#2a78d6"; GRAY = "#898781"; NAVY = "#101073"; RED = "#C81919"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 240,
                     "savefig.bbox": "tight"})


def draw_matrix(ax, A, diag_color, off_color, values=False, fontsize=7.0):
    """One cell per entry: filled where A is non-zero, faint outline where it is not."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if v == 0:
                ax.add_patch(Rectangle((j, n-1-i), 1, 1, facecolor="white",
                                       edgecolor="#e4e4e8", lw=0.6))
                continue
            col = diag_color if i == j else off_color
            ax.add_patch(Rectangle((j, n-1-i), 1, 1, facecolor=col, edgecolor="white",
                                   lw=0.8, alpha=1.0 if i == j else 0.72))
            if values:
                ax.text(j+0.5, n-1-i+0.5, f"{v:g}", ha="center", va="center",
                        color="white", fontsize=fontsize, fontweight="bold" if i == j else "normal")
    ax.set_xlim(-0.15, n+0.15); ax.set_ylim(-0.15, n+0.15)
    ax.set_aspect("equal"); ax.axis("off")


def chain_matrix(n, diag=2.5):
    A = np.zeros((n, n))
    np.fill_diagonal(A, diag)
    for i in range(n-1):
        A[i, i+1] = A[i+1, i] = -1.0
    return A


def grid_matrix(mx, my, screening=0.0):
    """Five-point stencil on an mx-by-my tiling, unknowns numbered row by row."""
    n = mx*my
    A = np.zeros((n, n))
    idx = lambda i, j: j*mx + i
    for j in range(my):
        for i in range(mx):
            k = idx(i, j)
            A[k, k] = 4.0 + screening
            if i+1 < mx: A[k, idx(i+1, j)] = A[idx(i+1, j), k] = -1.0
            if j+1 < my: A[k, idx(i, j+1)] = A[idx(i, j+1), k] = -1.0
    return A


# ------------------------------------------------------------------ 1. chain
n = 9
A = chain_matrix(n)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
draw_matrix(ax, A, GREEN, GREEN, values=True, fontsize=8.0)
ax.set_title("chain — one row of tiles:  $A$ is $\\bf{tridiagonal}$\n"
             f"{n} unknowns, {int((A != 0).sum())} non-zeros",
             fontsize=12, color="#222", pad=10)
ax.text(n/2, -0.75, "row $i$ touches $i-1$ and $i+1$ — and nothing else",
        ha="center", va="top", fontsize=10.5, color=GRAY)
fig.savefig("/tmp/mat/matrix-chain.png", transparent=True)
plt.close(fig)

# ------------------------------------------------------------- 2. 2-D lattice
mx, my = 5, 4
A = grid_matrix(mx, my)
n = mx*my
fig, ax = plt.subplots(figsize=(6.0, 5.0))
draw_matrix(ax, A, BLUE, BLUE)
ax.set_title("2-D lattice — the whole die:  $A$ is the $\\bf{five}$-$\\bf{point}$ $\\bf{stencil}$\n"
             f"{mx}×{my} tiles, {n} unknowns, {int((A != 0).sum())} non-zeros",
             fontsize=12, color="#222", pad=10)

# the two bands, named — labels parked to the right of the matrix
ax.set_xlim(-0.15, n + 10.5)
def band_label(i, j, text, dy):
    x, y = j + 0.5, n - 0.5 - i
    ax.annotate("", xy=(x + 0.45, y), xytext=(n + 1.2, y + dy),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.5,
                                connectionstyle="arc3,rad=-0.15", shrinkA=2, shrinkB=2))
    ax.text(n + 1.4, y + dy, text, color=NAVY, fontsize=10.5, ha="left", va="center")
band_label(3, 4,  "$\\pm 1$:  the tile beside it", dy=2.4)
band_label(9, 14, "$\\pm m$:  the tile above or below", dy=-2.4)

# the tear in the first off-diagonal, where a row of the die ends
for k in range(mx-1, n-1, mx):
    ax.add_patch(Rectangle((k+1, n-1-k), 1, 1, facecolor="none", edgecolor=RED, lw=2.0))
    ax.add_patch(Rectangle((k, n-1-(k+1)), 1, 1, facecolor="none", edgecolor=RED, lw=2.0))
ax.text(n/2, -0.75, "red: where a row of tiles ends — there the neighbour is not $i\\pm1$",
        ha="center", va="top", fontsize=10.5, color=RED)
fig.savefig("/tmp/mat/matrix-lattice.png", transparent=True)
plt.close(fig)
print("wrote matrix-chain.png and matrix-lattice.png")
