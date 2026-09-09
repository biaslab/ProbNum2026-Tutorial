# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""A itself, for the experiment slide: the 24x24 tiling, coloured by value.

    red = 4 + c on the diagonal     blue = -1, one per neighbour     white = 0

At 576 x 576 the five bands are thinner than a pixel on a projector, so the
top-left corner is repeated as an inset at readable size.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RED = (0.784, 0.098, 0.098)      # C81919
BLUE = (0.165, 0.471, 0.839)     # 2A78D6
NAVY = "#101073"
M, C = 24, 0.4                   # the tiling of the experiment: n = 576
ZOOM = 30                        # inset window: wide enough to hold the +/- M band

n = M * M
A = np.zeros((n, n))
idx = lambda i, j: j * M + i
for j in range(M):
    for i in range(M):
        k = idx(i, j)
        A[k, k] = 4.0 + C
        if i + 1 < M: A[k, idx(i + 1, j)] = A[idx(i + 1, j), k] = -1.0
        if j + 1 < M: A[k, idx(i, j + 1)] = A[idx(i, j + 1), k] = -1.0

def rgb(block):
    img = np.ones(block.shape + (3,))
    img[block == -1.0] = BLUE
    img[block == 4.0 + C] = RED
    return img

fig, ax = plt.subplots(figsize=(3.5, 3.8))
ax.imshow(rgb(A), interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color("#b9b9c2"); sp.set_linewidth(0.8)
ax.set_title(f"$A$ for the {M}×{M} tiling:  {n} × {n},\n"
             f"{int((A != 0).sum())} non-zeros — five a row",
             fontsize=11, color=NAVY, pad=7)

# the same matrix, top-left corner, at a size the eye can resolve
inset = ax.inset_axes([0.50, 0.50, 0.48, 0.44], xlim=(-0.5, ZOOM - 0.5),
                      ylim=(ZOOM - 0.5, -0.5), xticks=[], yticks=[])
inset.imshow(rgb(A[:ZOOM, :ZOOM]), interpolation="nearest")
for k in range(ZOOM + 1):
    inset.axhline(k - 0.5, color="#dcdce4", lw=0.3)
    inset.axvline(k - 0.5, color="#dcdce4", lw=0.3)
for sp in inset.spines.values():
    sp.set_color(NAVY); sp.set_linewidth(1.2)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((-0.5, -0.5), ZOOM, ZOOM, facecolor="none",
                       edgecolor=NAVY, lw=1.2))                 # what the inset shows
inset.set_title(f"the first {ZOOM} rows, magnified", fontsize=9, color=NAVY, pad=3)

fig.legend(handles=[Patch(facecolor=RED, label="$4+c$"),
                    Patch(facecolor=BLUE, label="$-1$"),
                    Patch(facecolor="white", edgecolor="#c8c8d0", label="$0$")],
           loc="lower center", ncol=3, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(bottom=0.12)
fig.savefig("/tmp/claude-0/-home-claude/8bb626c0-8b55-5a95-a8ba-7ca4323e8db9/scratchpad/fig14/matrix-A.png",
            dpi=330, transparent=True, bbox_inches="tight", pad_inches=0.03)
print("n =", n, " nnz =", int((A != 0).sum()))
