# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""One matrix, three classical methods — what each of them inverts.

Companion to the convergence experiment on the "Experiment" slide. The sparsity
pattern is coloured by VALUE (4+c on the diagonal, -1 for each neighbour, 0
elsewhere); the highlight shows which entries the method solves with (M) and
which it only multiplies by (N), for the splitting A = M - N.

    Jacobi        M = D          Gauss-Seidel  M = D + L        CG   no splitting
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

RED = "#C81919"; BLUE = "#2a78d6"; NAVY = "#101073"; GRAY = "#898781"
GRID = "#e6e6ea"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 240,
                     "savefig.bbox": "tight"})

MX, MY, C = 5, 4, 0.4                      # 5x4 tiling -> n = 20 unknowns


def grid_matrix(mx, my, c=0.0):
    n = mx * my
    A = np.zeros((n, n))
    idx = lambda i, j: j * mx + i
    for j in range(my):
        for i in range(mx):
            k = idx(i, j)
            A[k, k] = 4.0 + c
            if i + 1 < mx: A[k, idx(i + 1, j)] = A[idx(i + 1, j), k] = -1.0
            if j + 1 < my: A[k, idx(i, j + 1)] = A[idx(i, j + 1), k] = -1.0
    return A


def draw(ax, A, solved, title, subtitle, dim=0.14):
    """solved[i,j] = True where the method inverts the entry; the rest is dimmed."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            y = n - 1 - i
            if A[i, j] == 0:
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor="white",
                                       edgecolor=GRID, lw=0.5))
                continue
            col = RED if i == j else BLUE
            a = 1.0 if solved[i, j] else dim
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=col, edgecolor="white",
                                   lw=0.6, alpha=a))
    ax.set_xlim(-0.2, n + 0.2); ax.set_ylim(-0.2, n + 0.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=13, color=NAVY, pad=8, fontweight="bold")
    ax.text(n / 2, -1.0, subtitle, ha="center", va="top", fontsize=10.5, color="#333")


def staircase(ax, n, kind):
    """outline the region the method solves with: 'diag' for D, 'lower' for D + L"""
    if kind == "diag":
        for k in range(n):
            ax.add_patch(Rectangle((k, n - 1 - k), 1, 1, facecolor="none",
                                   edgecolor=NAVY, lw=1.4))
    else:
        pts = [(0, n), (1, n)]                       # close the region so it reads as one block
        for k in range(1, n):
            pts += [(k, n - k), (k + 1, n - k)]
        pts += [(n, 0), (0, 0), (0, n)]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=NAVY, lw=1.8, solid_joinstyle="miter", clip_on=False)


A = grid_matrix(MX, MY, C)
n = A.shape[0]
D = np.eye(n, dtype=bool)
LOW = np.tril(np.ones((n, n), dtype=bool))
ALL = np.ones((n, n), dtype=bool)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))

draw(axes[0], A, D, "Jacobi",
     "solves with the diagonal only\n$M = D$        >130 iterations")
staircase(axes[0], n, "diag")

draw(axes[1], A, LOW, "Gauss–Seidel",
     "solves with the lower triangle\n$M = D + L$        110 iterations")
staircase(axes[1], n, "lower")

draw(axes[2], A, ALL, "Conjugate gradients",
     "no splitting: all of $A$, via $Av$\n+ 2 global inner products      41 iterations")
axes[2].add_patch(Rectangle((-0.35, -0.35), n + 0.7, n + 0.7, facecolor="none",
                            edgecolor=NAVY, lw=1.8, linestyle=(0, (4, 2)), clip_on=False))

legend = [Patch(facecolor=RED, label="$A_{ii} = 4+c$"),
          Patch(facecolor=BLUE, label="$A_{ij} = -1$  (a neighbour)"),
          Patch(facecolor="white", edgecolor=GRID, label="$A_{ij} = 0$"),
          Patch(facecolor=BLUE, alpha=0.14, label="pale = only multiplied by ($N$)")]
fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, -0.10))
fig.suptitle("one matrix, three ways of using it   ($A = M - N$; a 5×4 tiling drawn here, the experiment runs 24×24)",
             fontsize=11.5, color=GRAY, y=1.04)
fig.savefig("/tmp/claude-0/-home-claude/8bb626c0-8b55-5a95-a8ba-7ca4323e8db9/scratchpad/fig14/splitting-three-methods.png",
            transparent=True)
print("written")
