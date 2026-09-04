### A Pluto.jl notebook ###
# v1.0.3

#> [frontmatter]
#> title = "Linear systems by message passing"
#> date = "2026-08-24"
#> tags = ["probabilistic numerics", "belief propagation", "linear solvers", "distributed inference"]
#> description = "ProbNum 2026 tutorial: solving Ax = b as marginal inference in a Gaussian Markov random field, and the classical iterative solvers as message passing with the uncertainty deleted."

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000001
begin
	using Plots
	using PlutoUI
	using LinearAlgebra
	using SparseArrays
	using Random
	using Printf
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000002
md"""
# Solving systems of equations with distributed probabilistic numerics

**ProbNum 2026 tutorial — Julia / Pluto edition**

Probabilistic numerics turns a computation into an inference problem. That move is by now familiar for linear solvers: a Krylov method is a Gaussian agent that has seen ``k`` matrix–vector products and reports a belief about ``x_\ast = A^{-1}b``. It is also, at scale, expensive: the belief lives on all of ``\mathbb{R}^n``, its covariance is dense, and every iteration needs global inner products — a synchronisation barrier across the whole machine.

This tutorial takes a different route to the same destination. Instead of treating ``A`` as a black box we can only probe, we read its **sparsity pattern as a graphical model**. Solving ``Ax = b`` then becomes *marginal inference* in a Gaussian Markov random field, and the natural algorithm is not conditioning-on-projections but **message passing**: every unknown is an agent, every non-zero ``A_{ij}`` is a channel, and the solver is a conversation between neighbours.

Nothing is global. No inner products, no barriers, no dense covariance. The belief is *local and anytime*: after ``k`` rounds every node holds a distribution built from exactly the information that has reached it.

| | |
|:--|:--|
| **1. Problem specification** | large sparse systems, and why "large" forces "distributed" |
| **2. Classical numerical approach** | direct factorisation, Jacobi/Gauss–Seidel, conjugate gradients |
| **3. Probabilistic numerical approach** | ``Ax=b`` as the mean of ``\mathcal{N}(A^{-1}b, A^{-1})`` |
| **4. Message-passing version** | Gaussian belief propagation, and what it costs |
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000003
md"""
!!! info "Where this comes from"
	The construction below is the Gaussian belief propagation (GaBP) solver of Shental, Bickson, Siegel, Wolf & Dolev, *Gaussian belief propagation solver for systems of linear equations* (ISIT 2008; extended version [arXiv:0810.1119](https://arxiv.org/abs/0810.1119)), together with its interior-point descendant (Bickson et al., *Polynomial linear programming with Gaussian belief propagation*, Allerton 2008) and its non-symmetric extension (Fanaskov, *Gaussian belief propagation solvers for nonsymmetric systems of linear equations*, SIAM J. Sci. Comput. 2022).

	This is the Julia edition of `python/01-linear-systems-by-message-passing.py`; the two notebooks compute the same things and agree to the digits shown.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000004
TableOfContents(; depth = 2)

# ╔═╡ aabbccdd-0000-4000-8000-000000000005
md"""
# 1. Problem specification

### Why would anyone solve ``Ax = b``?

Take a tiled accelerator: a few thousand compute tiles on one die, each with its own power counters, its own temperature sensor, and its own ability to throttle its clock. The chip is thermally interesting. A tile that runs hot has to slow down — but so, often, do its neighbours, because heat spreads sideways through the silicon faster than any of them can react. To decide who throttles, the control loop needs the **steady-state temperature of every tile**, and it needs it refreshed faster than the die's thermal time constant.

The physics fits in one line. Tile ``i`` dissipates power ``b_i``; it conducts heat to the four tiles it touches; and it loses heat to the coolant at a rate proportional to how far above ambient it sits. In steady state the three terms balance:

```math
\underbrace{c\,x_i}_{\text{lost to the coolant}} \;+\; \underbrace{\sum_{j \sim i}\,(x_i - x_j)}_{\text{conducted to neighbours}} \;=\; \underbrace{b_i}_{\text{dissipated on tile } i}.
```

That is one equation per tile, each involving five unknowns, and it is exactly the five-point discretisation of the screened Poisson equation ``(c - \Delta)u = f``. Stack the tile temperatures into ``x`` and the dissipated powers into ``b`` and the control loop is asking for

```math
A x = b, \qquad A \in \mathbb{R}^{n\times n}\ \ \text{symmetric},\ \text{sparse},\ \text{one row per tile.}
```

Keep the die in mind, because three features of it are what this tutorial is really about — and they are not special to silicon. The same three hold for a power grid, a sensor network, a robot swarm and a domain-decomposed PDE; the die is just the case where they are hardest to argue with.

* **``A`` is sparse and structured — it *is* the floorplan.** Row ``i`` couples ``x_i`` to the handful of tiles it physically touches, and to nothing else. The graph of the matrix is the layout of the chip. (Elsewhere it is a discretised differential operator, a Gauss–Markov model, a network of sensors — same picture.)
* **The data is already distributed.** ``b_i`` is a number tile ``i``'s own counters measured, and row ``i`` of ``A`` is a property of tile ``i``'s own package. Nothing was ever assembled anywhere. Assembling it means shipping every tile's telemetry to one place — every control period, forever.
* **Global synchronisation is the bottleneck.** With a few thousand tiles, a barrier costs more than the arithmetic between barriers, and by the time everyone has checked in, the temperature field has moved. Krylov methods need two inner products per step: that is two barriers per step.

So ``n`` is large enough that the interesting quantity is not "how many flops" but **how the work is laid out across the machine** — and, since we are at a probabilistic-numerics meeting, what each tile is entitled to believe while the answer is still arriving.

Our two running examples are the topologies that bracket the difficulty:

* a **chain** — one row of tiles, a tridiagonal system, whose graph is a *tree*;
* a **2-D lattice** — the whole die, a five-point stencil for ``(c - \Delta)u = f``, whose graph is *loopy*.

The parameter ``c \ge 0`` is the screening (reaction) term — here, the strength of the coupling to the coolant. It already has a physical meaning: it sets how far a hotspot is felt. A well-cooled die screens each hot tile into a small halo; a poorly cooled one lets hotspots talk to each other across the chip. It has a probabilistic meaning too, which we will come back to: ``c`` sets the **correlation length** of the associated Gaussian field, and with it everything about how far information has to travel.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000006
begin
	"Tridiagonal system: the graph of A is a chain, i.e. a tree."
	chain_matrix(n; diag = 2.5) =
		spdiagm(-1 => -ones(n - 1), 0 => fill(diag, n), 1 => -ones(n - 1))

	"Five-point stencil for (c − Δ) on an m×m lattice: the graph is loopy."
	function grid_matrix(m; screening = 0.0)
		d = 4.0 + screening
		T = spdiagm(-1 => -ones(m - 1), 0 => fill(d, m), 1 => -ones(m - 1))
		band = spdiagm(-1 => -ones(m - 1), 1 => -ones(m - 1))
		kron(sparse(I, m, m), T) + kron(band, sparse(I, m, m))
	end

	"Two localised sources on the m×m lattice — a hot core and a cooling channel."
	function bump_forcing(m; centers = ((0.3, 0.35), (0.7, 0.65)), width = 0.12)
		g = ((0:m-1) .+ 0.5) ./ m
		f = zeros(m, m)
		for (k, (cx, cy)) in enumerate(centers), i in 1:m, j in 1:m
			f[i, j] += (isodd(k) ? 1.0 : -0.8) *
				exp(-((g[i] - cx)^2 + (g[j] - cy)^2) / (2 * width^2))
		end
		vec(permutedims(f))              # row-major flatten: node (i,j) ↦ (i-1)m + j
	end

	"Undo the row-major flatten, for plotting."
	togrid(v, m) = permutedims(reshape(v, m, m))

	PAL = (blue = "#2a78d6", black = "#0b0b0b", green = "#008300", orange = "#eb6834",
	       pink = "#e87ba4", gray = "#898781", violet = "#4a3aa7", white = "#fcfcfb")
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000007
md"""
The picture to keep in mind for the rest of the tutorial: **the matrix *is* a graph**. Node ``i`` is the unknown ``x_i``; there is an edge ``\{i,j\}`` whenever ``A_{ij} \neq 0``. Everything the solver will do is expressible as nodes talking along those edges.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000008
let
	m = 6
	A = grid_matrix(m)
	Ao = A - spdiagm(0 => diag(A))
	rows, cols, _ = findnz(Ao)
	xy = [((i - 1) ÷ m, (i - 1) % m) for i in 1:m^2]

	plt = plot(; aspect_ratio = :equal, legend = :top, grid = false,
		axis = false, ticks = false, size = (620, 460),
		title = "The graph of A: $(m)×$(m) lattice, $(nnz(A)) non-zeros")
	first_edge = true
	for e in eachindex(rows)
		rows[e] < cols[e] || continue
		p, q = xy[rows[e]], xy[cols[e]]
		plot!(plt, [p[2], q[2]], [-p[1], -q[1]]; color = PAL.gray, lw = 1.5,
			label = first_edge ? "edges  (Aᵢⱼ ≠ 0)" : "")
		first_edge = false
	end
	scatter!(plt, [p[2] for p in xy], [-p[1] for p in xy];
		color = PAL.blue, ms = 9, msw = 1.5, msc = PAL.white, label = "unknowns  xᵢ")
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000009
md"""
# 2. Classical numerical approach

Three families, three different bargains.

**Direct solvers.** Factorise ``A = LL^\top`` and substitute. Exact in exact arithmetic, and for the 2-D lattice the factor ``L`` suffers *fill-in*: a banded matrix with ``O(n)`` non-zeros produces a factor with ``O(n^{3/2})`` of them. The graph view explains why — eliminating a node connects all of its neighbours to each other, so the graph densifies as you go.

**Stationary iterative methods.** Split ``A = D + R`` and iterate

```math
x^{(t+1)} = D^{-1}\bigl(b - R\,x^{(t)}\bigr) \qquad \text{(Jacobi)},
```

i.e. *each unknown solves its own equation, assuming its neighbours are right*. This is already local and distributed, as node ``i`` only needs ``x_j`` for its neighbours. Gauss–Seidel is the same update with values used as soon as they are available (asynchronous rather than synchronous). Both are simple and both converge slowly, at a rate set by the spectral radius of the iteration matrix.

**Krylov methods.** Conjugate gradients minimises the ``A``-norm error over the Krylov space and converges in ``O(\sqrt{\kappa}\,\log \varepsilon^{-1})`` iterations, which is far better. The price is that every step needs ``r^\top r`` and ``p^\top Ap``, two inner products over all ``n`` entries. On a distributed machine every processor must wait for every other.

All three return a vector ``x_k`` plus an error bound in terms of quantities (``\kappa``, ``\|x_\ast\|``) that are exactly as unknown as the solution.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000010
begin
	"Jacobi (kind = :jacobi) or Gauss–Seidel (kind = :gs) iterates, from x = 0."
	function stationary(A, b, iters; kind = :jacobi)
		d = diag(A)
		x = zeros(length(b))
		out = [copy(x)]
		for _ in 1:iters
			if kind == :jacobi
				x = x + (b - A * x) ./ d           # every node solves its own row
			else
				for i in eachindex(x)              # ... using whatever has arrived
					x[i] += (b[i] - dot(A[i, :], x)) / d[i]
				end
			end
			push!(out, copy(x))
		end
		out
	end

	"Textbook CG from x = 0, returning every iterate."
	function conjugate_gradients(A, b, iters)
		x = zeros(length(b))
		r = b - A * x
		p = copy(r)
		rr = dot(r, r)
		out = [copy(x)]
		for _ in 1:iters
			Ap = A * p
			α = rr / dot(p, Ap)
			x = x + α * p
			r = r - α * Ap
			rr_new = dot(r, r)
			p = r + (rr_new / rr) * p
			rr = rr_new
			push!(out, copy(x))
		end
		out
	end
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000011
md"""
# 3. Probabilistic numerical approach

The usual probabilistic reading of a linear solver puts a Gaussian prior on ``x``, treats each matrix–vector product as a linear observation ``s_i^\top A x_\ast = s_i^\top b``, and conditions. It inherits a dense ``n \times n`` covariance and a global policy for choosing ``s_i``.

For symmetric positive definite ``A``, define

```math
q(x) = \tfrac12 x^\top A x - b^\top x, \qquad
p(x) \;\propto\; \exp\bigl(-q(x)\bigr) \;=\; \exp\bigl(-\tfrac12 x^\top A x + b^\top x\bigr).
```

Completing the square gives

```math
p(x) \;=\; \mathcal{N}\bigl(x;\ A^{-1}b,\ A^{-1}\bigr).
```

So the solution vector *is* the mean of a Gaussian whose **precision matrix is ``A`` itself** (Shental et al. 2008, Prop. 8). Solving a linear system and computing the marginal means of a Gaussian Markov random field are the same problem:

| linear algebra | probabilistic inference |
|:--|:--|
| matrix ``A`` | precision (information) matrix |
| right-hand side ``b`` | natural-parameter mean ``A\mu`` |
| sparsity pattern of ``A`` | conditional independence graph |
| solution ``x_\ast = A^{-1}b`` | vector of marginal means ``\mu_i`` |
| diagonal of ``A^{-1}`` | marginal variances ``\sigma_i^2`` |
| ``1/A_{ii}`` | *conditional* variance of ``x_i`` given its neighbours |

This leads us to two observations.

1. **The uncertainty is free-standing.** We did not choose a prior and we are not modelling rounding error. The Gaussian is a re-encoding of the problem itself, and its marginal variances ``(A^{-1})_{ii}`` are the quantity a statistician would want anyway when ``A`` is a posterior precision (Gaussian process regression, GMRF models, Kalman smoothing, bundle adjustment).

   Back on the die, that quantity is not a statistical abstraction either: ``(A^{-1})_{ij}`` is the temperature rise at tile ``i`` per unit of power dissipated at tile ``j`` — the **thermal impedance** of the chip, which is what a thermal engineer would have measured. Its diagonal, the marginal variance, is how hot tile ``i`` gets from its own watt *once the rest of the die has been allowed to warm up in response*. The conditional variance ``1/A_{ii}`` is the same number computed with every neighbour pinned to ambient: the answer a tile would give if it believed it were the only warm thing on the chip. The gap between those two is real, physical, and — as §4 will show — is precisely what the messages carry.

2. **Locality is now structural.** ``A_{ij} = 0`` means ``x_i \perp x_j \mid x_{\text{rest}}``. The graph of the matrix is the conditional independence graph of the belief, so an inference algorithm that only exchanges information along edges is *automatically* a solver that only communicates along the sparsity pattern.

The last row of the table is the seed of the entire algorithm. A node that knows only its own equation knows the conditional variance ``1/A_{ii}``; to upgrade it to the marginal variance ``(A^{-1})_{ii}`` it has to hear from the rest of the graph.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000012
let
	A = [3.0 1.4; 1.4 2.0]
	b = [1.0, 2.0]
	x = A \ b
	g = range(-1.2, 1.8; length = 160)
	q(u, v) = 0.5 * (A[1,1]*u^2 + 2*A[1,2]*u*v + A[2,2]*v^2) - (b[1]*u + b[2]*v)

	plt = contourf(g, g, (u, v) -> exp(-(q(u, v) - q(x[1], x[2])));
		levels = 18, linewidth = 0, color = :blues, colorbar = false,
		aspect_ratio = :equal, size = (560, 440),
		xlabel = "x₁", ylabel = "x₂", legend = :topleft,
		title = "p(x) ∝ exp(−½ xᵀA x + bᵀx) = 𝒩(A⁻¹b, A⁻¹)")
	scatter!(plt, [x[1]], [x[2]]; m = :star5, ms = 11, color = PAL.orange,
		msc = PAL.white, msw = 1, label = "solution = mean  A⁻¹b")
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000013
md"""
# 4. The message-passing version

## 4.1 The factor graph

Write the density as a product of one factor per node and one per edge:

```math
p(x) \;\propto\; \prod_{i} \phi_i(x_i) \prod_{\{i,j\}} \psi_{ij}(x_i,x_j),
\qquad
\phi_i(x_i) = \exp\!\bigl(b_i x_i - \tfrac12 A_{ii} x_i^2\bigr),
\qquad
\psi_{ij}(x_i,x_j) = \exp(-x_i A_{ij} x_j).
```

The self-factor ``\phi_i`` is row ``i``'s own equation — it is ``\mathcal{N}(x_i;\, b_i/A_{ii},\, 1/A_{ii})``, exactly the "solve my equation ignoring the coupling" belief that Jacobi starts from. The edge factor ``\psi_{ij}`` is the coupling. **Every quantity in the factor graph is an entry of ``A`` or ``b`` that node ``i`` already owns.**

## 4.2 The messages

Sum-product on this graph: the message from ``i`` to ``j`` is

```math
m_{i\to j}(x_j) \;\propto\; \int \psi_{ij}(x_i,x_j)\, \phi_i(x_i) \!\!\prod_{k \in N(i)\setminus j}\!\! m_{k\to i}(x_i)\, \mathrm{d}x_i .
```

Products of Gaussians are Gaussian and Gaussian integrals are Gaussian, so each message is carried by **two scalars**: a precision ``P_{ij}`` and a mean ``\mu_{ij}``. Writing ``P_{i\setminus j}`` for the precision node ``i`` has accumulated *excluding* what ``j`` told it,

```math
P_{i\setminus j} = A_{ii} + \!\!\sum_{k\in N(i)\setminus j}\!\! P_{ki},
\qquad
\mu_{i\setminus j} = \frac{1}{P_{i\setminus j}}\Bigl(b_i + \!\!\sum_{k\in N(i)\setminus j}\!\! P_{ki}\mu_{ki}\Bigr),
```

the outgoing message is

```math
P_{ij} = -\frac{A_{ij}^2}{P_{i\setminus j}}, \qquad
\mu_{ij} = \frac{P_{i\setminus j}\,\mu_{i\setminus j}}{A_{ij}}
```

and the belief at node ``i``, using *all* incoming messages, is

```math
P_i = A_{ii} + \sum_{k\in N(i)} P_{ki},
\qquad
\mu_i = \frac{1}{P_i}\Bigl(b_i + \sum_{k\in N(i)} P_{ki}\mu_{ki}\Bigr),
\qquad
x_i \approx \mu_i, \quad (A^{-1})_{ii} \approx 1/P_i .
```

Note the sign: ``P_{ij} = -A_{ij}^2 / P_{i\setminus j}`` is **negative**. Messages are not probability distributions — they are *information updates*, and what a neighbour tells you here is "you are less certain than you thought": each message pushes a node's belief from the conditional variance ``1/A_{ii}`` towards the marginal variance ``(A^{-1})_{ii} \ge 1/A_{ii}``.

Everything on the right-hand side is indexed by ``i`` and its neighbours. There is no ``n`` anywhere in the update.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000014
"""
Directed off-diagonal edges of a structurally symmetric sparse matrix.

Returns `(src, dst, a, rev)`: `a[e] = A[src[e], dst[e]]`, and `rev[e]` is the index of
the reverse edge of `e` — the only bookkeeping the "exclude what j told me" rule needs.
"""
function edge_list(A::SparseMatrixCSC)
	Ao = dropzeros(A - spdiagm(0 => diag(A)))
	src, dst, a = findnz(Ao)
	pos = Dict((src[e], dst[e]) => e for e in eachindex(src))
	rev = [pos[(dst[e], src[e])] for e in eachindex(src)]
	src, dst, a, rev
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000015
"""
Gaussian belief propagation for A x = b  (Shental et al. 2008, Algorithms 1–2).

Messages live on directed edges and carry two scalars: a precision `P` and a
precision-weighted mean `W = P·μ`.  Every update touches one node and its neighbours only.

* `schedule = :parallel` — flooding; all nodes send simultaneously (à la Jacobi)
* `schedule = :serial`   — sweep nodes, using messages as soon as they arrive (à la Gauss–Seidel)
* `jacobi = true`        — clamp the precision messages to zero, which *is* Jacobi (Prop. 16)
"""
function gabp(A, b; iters = 500, tol = 1e-10, damping = 0.0,
              schedule = :parallel, jacobi = false, record = false)
	n = size(A, 1)
	Pii = Vector{Float64}(diag(A))          # self-factor precision
	Wii = Vector{Float64}(b)                # self-factor  P·μ  =  A_ii · (b_i/A_ii)
	src, dst, a, rev = edge_list(A)
	ne = length(a)
	P, W = zeros(ne), zeros(ne)             # message precision, and precision × mean
	bnorm = norm(b)
	inbox  = [findall(==(i), dst) for i in 1:n]
	outbox = [findall(==(i), src) for i in 1:n]

	res, mus, sds = Float64[], Vector{Float64}[], Vector{Float64}[]
	SP, SW, mu = copy(Pii), copy(Wii), zeros(n)
	for _ in 1:iters
		if schedule == :parallel
			SP .= Pii; SW .= Wii
			for e in 1:ne
				SP[dst[e]] += P[e]; SW[dst[e]] += W[e]
			end
			Pn, Wn = similar(P), similar(W)
			for e in 1:ne
				P_ex = SP[src[e]] - (jacobi ? 0.0 : P[rev[e]])   # exclude what j told i
				W_ex = SW[src[e]] - (jacobi ? 0.0 : W[rev[e]])
				Pn[e] = jacobi ? 0.0 : -a[e]^2 / P_ex
				Wn[e] = -a[e] * (W_ex / P_ex)
			end
			@. P = (1 - damping) * Pn + damping * P
			@. W = (1 - damping) * Wn + damping * W
		else
			for i in 1:n
				sp = Pii[i] + sum(@view P[inbox[i]]; init = 0.0)
				sw = Wii[i] + sum(@view W[inbox[i]]; init = 0.0)
				for e in outbox[i]
					P_ex = sp - (jacobi ? 0.0 : P[rev[e]])
					W_ex = sw - (jacobi ? 0.0 : W[rev[e]])
					Pn = jacobi ? 0.0 : -a[e]^2 / P_ex
					Wn = -a[e] * (W_ex / P_ex)
					P[e] = (1 - damping) * Pn + damping * P[e]
					W[e] = (1 - damping) * Wn + damping * W[e]
				end
			end
		end

		SP .= Pii; SW .= Wii
		for e in 1:ne
			SP[dst[e]] += P[e]; SW[dst[e]] += W[e]
		end
		mu = SW ./ SP
		r = norm(A * mu - b) / bnorm
		push!(res, r)
		if record
			push!(mus, copy(mu)); push!(sds, sqrt.(abs.(1.0 ./ SP)))
		end
		if !isfinite(r) || r > 1e10
			return (; mu, var = 1.0 ./ SP, res, mus, sds,
			          iters = length(res), converged = false)
		end
		r < tol && break
	end
	(; mu, var = 1.0 ./ SP, res, mus, sds,
	   iters = length(res), converged = res[end] < tol)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000016
md"""
That is the whole solver: no factorisation, no inner products, no ``n``-dimensional linear algebra. The `inbox` sums stand in for what would be, on real hardware, each node summing its own mailbox.

## 4.3 Sanity check on a 3×3 system

The toy example from Shental et al. (their eq. 47) is deliberately nasty: symmetric but **indefinite**, so "the Gaussian" is not a probability distribution at all. The algebra does not care.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000017
begin
	A_toy = sparse([1.0 -2.0 3.0; -2.0 1.0 0.0; 3.0 0.0 1.0])
	b_toy = [-6.0, 0.0, 2.0]
	toy_bp = gabp(A_toy, b_toy; iters = 200)
	toy_exact = Matrix(A_toy) \ b_toy
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000018
md"""
| | ``x_1`` | ``x_2`` | ``x_3`` |
|:--|--:|--:|--:|
| GaBP after $(toy_bp.iters) rounds | $(@sprintf("%.6f", toy_bp.mu[1])) | $(@sprintf("%.6f", toy_bp.mu[2])) | $(@sprintf("%.6f", toy_bp.mu[3])) |
| `A \\ b` | $(@sprintf("%.6f", toy_exact[1])) | $(@sprintf("%.6f", toy_exact[2])) | $(@sprintf("%.6f", toy_exact[3])) |

Eigenvalues of ``A``: $(join([@sprintf("%.2f", v) for v in eigvals(Matrix(A_toy))], ", ")) — not positive definite, and yet exact.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000019
md"""
## 4.4 Trees: message passing *is* Gaussian elimination

If the graph of ``A`` has no cycles, belief propagation is exact — in the means *and* in the variances — after at most as many rounds as the diameter of the tree (and in practice sooner: the messages stop changing once information has crossed a correlation length, not the whole graph). Shental et al. (Prop. 14) make the correspondence precise: on a tree, the message sweep from the leaves inward performs exactly the row operations of Gaussian elimination (``P_{i\setminus j}`` is the updated pivot ``A_{ii} - \sum_l A_{li}^2/A_{ll}``), and reading off the marginals is forward substitution.

A tridiagonal system is the simplest instance: GaBP on a chain **is** the Thomas algorithm, re-derived as inference. Below, both the solution and the marginal variances ``(A^{-1})_{ii}`` come out to machine precision — and the variances are the diagonal of a dense inverse we never formed.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000020
begin
	n_chain = 80
	A_chain = chain_matrix(n_chain; diag = 2.5)
	b_chain = randn(Xoshiro(2026), n_chain)
	chain_bp = gabp(A_chain, b_chain; iters = 2000, tol = 1e-13)
	chain_x = Matrix(A_chain) \ b_chain
	chain_v = diag(inv(Matrix(A_chain)))
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000021
let
	sd = sqrt.(chain_bp.var)
	plt = plot(1:n_chain, chain_bp.mu; ribbon = 2 .* sd, fillalpha = 0.15,
		color = PAL.blue, lw = 2, label = "GaBP marginal means  (± 2σ)",
		xlabel = "node i", ylabel = "xᵢ", legend = :top, size = (680, 400),
		title = "Chain of $(n_chain) unknowns — converged in $(chain_bp.iters) rounds")
	plot!(plt, 1:n_chain, chain_x; color = PAL.black, lw = 1.5, ls = :dash,
		label = "exact solution")
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000022
md"""
| | max abs. error |
|:--|--:|
| means vs `A \\ b` | $(@sprintf("%.2e", maximum(abs.(chain_bp.mu .- chain_x)))) |
| variances vs `diag(inv(A))` | $(@sprintf("%.2e", maximum(abs.(chain_bp.var .- chain_v)))) |
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000023
md"""
## 4.5 What "belief" means before convergence

Run the message passing for ``k`` rounds and stop. What is node ``i`` holding?

The marginal of the **computation tree of depth ``k``** rooted at ``i``. This is the graph you get by unrolling the neighbourhood of ``i`` for ``k`` hops. That is the sub-problem whose information has physically reached node ``i`` in ``k`` rounds of communication. So the belief at iteration ``k`` is not a heuristic error estimate; it is the *exact posterior of the part of the problem the node has seen so far*, and the sequence interpolates from

```math
\text{iteration } 0:\quad \mathcal{N}\bigl(b_i/A_{ii},\; 1/A_{ii}\bigr)
\qquad\text{(the conditional: "my equation, neighbours assumed known")}
```

to

```math
\text{convergence}:\quad \mathcal{N}\bigl((A^{-1}b)_i,\; (A^{-1})_{ii}\bigr)
\qquad\text{(the marginal: the whole system accounted for).}
```

Each node's uncertainty therefore *grows* as information arrives. The early over-confidence of "I'll just solve my own row" is corrected by neighbours. The belief is **local and anytime**: every node has one at every round, computed from the messages it happens to hold, with no global quantity ever assembled.

In the story of §1: at round 0 every tile reports the temperature it would reach if it were the only warm thing on the die, and reports it with the confidence of the isolated. Round by round it learns that its neighbours are warm, that theirs are, and that it sits in a hot region of the chip — its estimate rises and its stated certainty falls. After ``k`` rounds a tile has accounted for exactly the ``k``-hop patch of silicon around it. **That patch is what its belief describes** — not, as the box below insists, how wrong its number is.

Watch the front of information sweep across the lattice.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000024
begin
	m_grid = 24                       # 24 × 24 lattice, n = 576 unknowns
	screen_grid = 0.4
	A_grid = grid_matrix(m_grid; screening = screen_grid)
	b_grid = bump_forcing(m_grid)
	x_grid = Matrix(A_grid) \ b_grid
	grid_bp = gabp(A_grid, b_grid; iters = 400, tol = 1e-12, record = true)
	var_grid = diag(inv(Matrix(A_grid)))         # reference marginals (dense, n = 576)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000025
md"""
message-passing rounds ``k``: $(@bind round_k Slider(1:min(60, grid_bp.iters), default = 3, show_value = true))
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000026
let
	μ = togrid(grid_bp.mus[round_k], m_grid)
	sd = togrid(grid_bp.sds[round_k], m_grid)
	sd_true = togrid(sqrt.(var_grid), m_grid)

	h1 = heatmap(μ; color = :RdBu, clims = (minimum(x_grid), maximum(x_grid)),
		yflip = true, aspect_ratio = :equal, axis = false, ticks = false,
		title = "belief mean μᵢ")
	h2 = heatmap(sd; color = :viridis,
		clims = (0.95 * minimum(sd_true), 1.02 * maximum(sd_true)),
		yflip = true, aspect_ratio = :equal, axis = false, ticks = false,
		title = "belief std √(1/Pᵢ)")
	plot(h1, h2; layout = (1, 2), size = (760, 360),
		plot_title = "round k = $(round_k)")
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000027
md"""
| after k = $(round_k) rounds | value |
|:--|--:|
| relative residual ‖Aμ − b‖/‖b‖ | $(@sprintf("%.2e", grid_bp.res[round_k])) |
| max error in the means | $(@sprintf("%.2e", maximum(abs.(grid_bp.mus[round_k] .- x_grid)))) |
| mean belief std (BP) | $(@sprintf("%.4f", sum(grid_bp.sds[round_k]) / length(x_grid))) |
| mean marginal std (exact) | $(@sprintf("%.4f", sum(sqrt.(var_grid)) / length(x_grid))) |
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000028
md"""
!!! tip "What to look for"
	At ``k=1`` every node reports ``b_i/A_{ii}`` — its own equation, nothing else — and a uniformly small standard deviation: maximal over-confidence. As rounds pass, the mean fills in from the sources outward, and the standard-deviation map inflates from the boundary inward, because nodes near the boundary genuinely *are* better determined (Dirichlet conditions pin them) while interior nodes must wait to learn how loosely they are held. Both fields stop changing once the information has travelled a correlation length.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000029
md"""
!!! warning "Do not over-read this"
	It is tempting — and this notebook's first draft did exactly that — to call the round-``k`` belief an *error bar on the computation*. It is not. It is the exact posterior of a **different problem**: the truncated ``k``-hop computation tree. Nothing in it estimates the distance between ``\mu^{(k)}`` and the answer ``A^{-1}b``.

	The reason is visible in the update rule itself. The precision recursion ``P_{ij} = -A_{ij}^2 / P_{i\setminus j}`` **contains no ``b``**: the precisions form a closed system driven by the matrix alone. Change the right-hand side and every variance in this notebook is unchanged to the last bit, while the errors are completely different. So the belief cannot be tracking the error, and the two converge on schedules that have nothing to do with each other.

	That gap — an anytime *belief* that is not an anytime *error estimate* — is, to us, the most interesting open problem in this whole construction, and §5 comes back to it.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000030
md"""
## 4.6 Loops: exact means, over-confident variances

On a graph with cycles the same information arrives at a node by several routes and gets double-counted. The remarkable fact (Weiss & Freeman 2001) is that this does **not** spoil the means: *if* GaBP converges, the marginal means are the exact solution ``A^{-1}b``, cycles or no cycles. The variances are another matter — the computation tree that BP effectively solves keeps re-entering the same loop, and the walk-sum analysis of Malioutov, Johnson & Willsky (2006) shows BP counts only the self-return walks that revisit the root once. On an attractive model, where all those walks contribute with the same sign, the missing terms are positive, so BP **under-estimates** the variance: the solver is over-confident.

That is the honest state of the art, and it is exactly the kind of statement the probabilistic-numerics community is equipped to improve on.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000031
let
	lo = min(minimum(grid_bp.var), minimum(var_grid))
	hi = max(maximum(grid_bp.var), maximum(var_grid))
	plt = plot([lo, hi], [lo, hi]; color = PAL.black, lw = 1.5, ls = :dash,
		label = "exact", legend = :topleft, size = (560, 420),
		xlabel = "(A⁻¹)ᵢᵢ  (exact)", ylabel = "1/Pᵢ  (belief propagation)",
		title = "Converged GaBP variances vs the true diagonal of A⁻¹")
	scatter!(plt, var_grid, grid_bp.var; color = PAL.blue, ms = 3.5, msw = 0.5,
		msc = PAL.white, alpha = 0.6, label = "one node")
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000032
md"""
| converged GaBP on the 24×24 lattice | |
|:--|--:|
| max error in the **means** | $(@sprintf("%.2e", maximum(abs.(grid_bp.mu .- x_grid)))) |
| ratio BP variance / true variance | $(@sprintf("%.3f", minimum(grid_bp.var ./ var_grid))) – $(@sprintf("%.3f", maximum(grid_bp.var ./ var_grid))) |
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000033
md"""
## 4.7 The punchline: Jacobi is GaBP with the uncertainty deleted

Take the algorithm above and make two changes:

1. clamp every precision message to zero, ``P_{ij} := 0``;
2. stop excluding the reverse message — let node ``i`` use what ``j`` told it when replying to ``j``.

What remains is ``\mu_i = A_{ii}^{-1}\bigl(b_i - \sum_{k \neq i} A_{ki}\mu_k\bigr)``: **the Jacobi iteration** (Shental et al., Prop. 16). The classical stationary solver is the message-passing solver with the second moment thrown away and the cycle-avoidance thrown away.

This is the clearest statement of the tutorial's thesis. The classical method is not an alternative to the probabilistic one; it is the probabilistic one, marginalised down to a point estimate. Everything GaBP does beyond Jacobi — carrying precisions, excluding the reverse message — is *bookkeeping about information*, and it is what buys both the uncertainty estimate and the faster convergence.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000034
begin
	K_jac = 40
	clamped = gabp(A_grid, b_grid; iters = K_jac, jacobi = true, tol = 0.0, record = true)
	jac_iterates = stationary(A_grid, b_grid, K_jac + 1; kind = :jacobi)
	# x⁰ = 0, so Jacobi's iterate k+2 is round k of the clamped message passing
	jac_gap = maximum(maximum(abs.(clamped.mus[k] .- jac_iterates[k + 2])) for k in 1:K_jac)
	full_bp = gabp(A_grid, b_grid; iters = K_jac, tol = 0.0)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000035
md"""
| | |
|:--|--:|
| max difference over $(K_jac) rounds between "GaBP with ``P_{ij} := 0``" and Jacobi | **$(@sprintf("%.2e", jac_gap))** |
| relative residual after $(K_jac) rounds — Jacobi | $(@sprintf("%.2e", norm(A_grid * jac_iterates[K_jac + 1] - b_grid) / norm(b_grid))) |
| relative residual after $(K_jac) rounds — full GaBP | $(@sprintf("%.2e", full_bp.res[end])) |

Identical to machine precision — and the two residuals show what the discarded second moment was worth.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000036
md"""
## 4.8 Scheduling: nobody has to wait

Message passing does not prescribe *when* nodes speak. Two standard choices:

* **flooding (parallel)** — every node sends every round, using the previous round's messages. Synchronous, like Jacobi.
* **serial (asynchronous)** — sweep the nodes and use each message the moment it exists. Like Gauss–Seidel, and typically about twice as fast.

Neither needs an inner product, a norm, or any other quantity that couples all ``n`` unknowns. Convergence is not destroyed by nodes running at different speeds, by stale messages, or by a node dropping out for a while — which is what makes the scheme viable on an unreliable, heterogeneous, or genuinely geographically distributed machine. Compare against the classical methods, remembering that each CG iteration hides two global barriers.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000037
begin
	K_cmp = 120
	residual_curve(A, b, xs) = [norm(A * x - b) / norm(b) for x in xs]

	curves_grid = [
		"Jacobi"              => residual_curve(A_grid, b_grid, stationary(A_grid, b_grid, K_cmp; kind = :jacobi)),
		"Gauss–Seidel"        => residual_curve(A_grid, b_grid, stationary(A_grid, b_grid, K_cmp; kind = :gs)),
		"conjugate gradients" => residual_curve(A_grid, b_grid, conjugate_gradients(A_grid, b_grid, K_cmp)),
		"GaBP (flooding)"     => [1.0; gabp(A_grid, b_grid; iters = K_cmp, tol = 1e-14).res],
		"GaBP (serial)"       => [1.0; gabp(A_grid, b_grid; iters = K_cmp, tol = 1e-14, schedule = :serial).res],
	]
	curves_chain = [
		"Jacobi"              => residual_curve(A_chain, b_chain, stationary(A_chain, b_chain, K_cmp; kind = :jacobi)),
		"conjugate gradients" => residual_curve(A_chain, b_chain, conjugate_gradients(A_chain, b_chain, K_cmp)),
		"GaBP (flooding)"     => [1.0; gabp(A_chain, b_chain; iters = K_cmp, tol = 1e-14).res],
	]
	curve_style = Dict(
		"Jacobi"              => (PAL.gray,   :dot),
		"Gauss–Seidel"        => (PAL.pink,   :dot),
		"conjugate gradients" => (PAL.orange, :dashdot),
		"GaBP (flooding)"     => (PAL.blue,   :solid),
		"GaBP (serial)"       => (PAL.green,  :solid),
	)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000038
md"""
problem: $(@bind problem_pick Select(["lattice" => "24×24 lattice (loopy)", "chain" => "chain of 80 (tree)"]))
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000039
let
	curves = problem_pick == "lattice" ? curves_grid : curves_chain
	plt = plot(; yscale = :log10, ylims = (1e-15, 3.0), size = (700, 430),
		xlabel = "iteration", ylabel = "relative residual", legend = :topright,
		title = "Relative residual ‖Ax − b‖ / ‖b‖ per iteration")
	for (name, c) in curves
		color, ls = curve_style[name]
		plot!(plt, 0:length(c)-1, max.(c, 1e-16); color, ls, lw = 2, label = name)
	end
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000040
md"""
!!! info "Reading the plot"
	GaBP sits between the stationary methods and CG in iteration count — clearly better than Jacobi, comparable to or better than Gauss–Seidel — while being *strictly more local than either*: no global norm is ever formed, and the serial variant tolerates arbitrary update order. CG wins on iterations; whether it wins on wall-clock depends entirely on what a global reduction costs you. And on the tree, GaBP terminates *exactly* — in a bounded number of rounds, set by how far information must travel — which no stationary method does.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000041
md"""
## 4.9 Does it scale?

Per round, each node sends one two-scalar message per incident edge: the cost is ``O(\mathrm{nnz})`` arithmetic and ``O(\mathrm{nnz})`` communication, all of it nearest-neighbour, all of it parallel. So the only question that matters is **how the round count grows with ``n``** — and that is where the probabilistic reading pays off in intuition.

The screening parameter ``c`` in ``(c - \Delta)u = f`` sets the correlation length ``\ell \sim 1/\sqrt{c}`` of the Gaussian field ``\mathcal{N}(A^{-1}b, A^{-1})``. A node's marginal is determined by the nodes within a few ``\ell`` of it; everything beyond is screened off. Information therefore has to travel a *fixed physical distance*, not across the whole domain — so the round count **saturates**: it stops growing with ``n``, and the total work is ``O(n)`` with perfect parallelism.

At ``c = 0`` the correlation length is the domain size, every node needs to hear from every other, and the round count grows like the diameter. This is not a defect of message passing; it is the same long-range coupling that makes unpreconditioned Jacobi and CG slow, showing up in probabilistic clothing — and it says precisely what a preconditioner has to do: *shorten the correlation length*.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000042
begin
	scale_sizes = [8, 12, 16, 24, 32, 48, 64]
	scale_screens = [0.0, 0.4, 2.0]
	scale_rounds = Dict(c => [gabp(grid_matrix(m; screening = c), bump_forcing(m);
	                              iters = 40_000, tol = 1e-8).iters
	                          for m in scale_sizes]
	                    for c in scale_screens)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000043
let
	plt = plot(; xscale = :log10, yscale = :log10, size = (700, 430), legend = :topleft,
		xlabel = "number of unknowns n", ylabel = "message-passing rounds",
		title = "Rounds to ‖Aμ − b‖/‖b‖ < 10⁻⁸, five-point stencil (c − Δ)")
	for (c, color) in zip(scale_screens, (PAL.orange, PAL.blue, PAL.green))
		ell = c == 0 ? "∞" : string(round(1 / sqrt(c); digits = 1))
		plot!(plt, scale_sizes .^ 2, scale_rounds[c]; color, lw = 2, marker = :circle,
			ms = 5, msc = PAL.white, label = "c = $(c)   (ℓ ≈ $(ell))")
	end
	plt
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000044
let
	hdr  = "| n | " * join(["rounds (c = $(c))" for c in scale_screens], " | ") * " |"
	sep  = "|---|" * repeat("---|", length(scale_screens))
	rows = ["| $(m^2) | " * join([string(scale_rounds[c][k]) for c in scale_screens], " | ") * " |"
	        for (k, m) in enumerate(scale_sizes)]
	note = "\nAcross a 64-fold increase in ``n``, the screened columns grow by a factor of two " *
	       "or less while the unscreened one grows with the diameter of the domain. " *
	       "A column that flattens is an ``O(n)`` solver with no global communication."
	Markdown.parse(join([hdr, sep, rows..., note], "\n"))
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000045
md"""
## 4.10 When it fails

GaBP is not unconditionally convergent, and the sufficient conditions are the familiar ones:

* ``A`` strictly **diagonally dominant** ``\Rightarrow`` convergence to the exact means (Weiss & Freeman 2001);
* **walk-summability**, ``\rho\bigl(|I - D^{-1}A|\bigr) < 1`` with ``D = \operatorname{diag}(A)``, a strictly weaker condition (Malioutov et al. 2006);
* a **tree**, in which case it converges exactly regardless of the spectral radius.

In practice the basin is considerably larger than those conditions — but it does have an edge. Below, a lattice with random ``\pm w`` couplings (a "frustrated" model, the sort where loops carry conflicting information) sweeps from harmless to divergent. Watch the diagnostics: both sufficient conditions are crossed long before anything goes wrong, and then convergence fails somewhere near the point where the Gaussian stops being a valid distribution at all.
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000046
"Lattice with random ±w couplings and unit diagonal — loops with conflicting information."
function frustrated_matrix(m, w; seed = 3)
	rng = Xoshiro(seed)
	G = grid_matrix(m) - spdiagm(0 => diag(grid_matrix(m)))
	rows, cols, _ = findnz(G)
	signs = Dict{Tuple{Int,Int},Float64}()
	vals = similar(rows, Float64)
	for e in eachindex(rows)
		key = minmax(rows[e], cols[e])
		vals[e] = w * get!(() -> rand(rng, (-1.0, 1.0)), signs, key)
	end
	sparse(rows, cols, vals, m^2, m^2) + sparse(I, m^2, m^2)
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000047
md"""
coupling strength ``w``: $(@bind coupling Slider(0.05:0.01:0.35, default = 0.20, show_value = true))

damping ``\alpha``: $(@bind damping_ui Slider(0.0:0.1:0.9, default = 0.0, show_value = true))
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000048
begin
	m_fr = 12
	A_fr = frustrated_matrix(m_fr, coupling)
	b_fr = randn(Xoshiro(7), m_fr^2)
	fr_run = gabp(A_fr, b_fr; iters = 600, tol = 1e-10, damping = damping_ui)

	fr_diag = let Ad = Matrix(A_fr), D = Diagonal(1 ./ diag(Matrix(A_fr)))
		(rho = maximum(abs.(eigvals(abs.(Matrix(1.0I, m_fr^2, m_fr^2) - D * Ad)))),
		 lam = minimum(eigvals(Symmetric(Ad))),
		 dd  = minimum(abs.(diag(Ad)) .- (vec(sum(abs.(Ad); dims = 2)) .- abs.(diag(Ad)))))
	end
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000049
let
	res = clamp.(replace(fr_run.res, NaN => 1e10), 1e-16, 1e10)
	plot(1:length(res), res; yscale = :log10, lw = 2, legend = false, size = (700, 340),
		color = fr_run.converged ? PAL.blue : PAL.orange,
		xlabel = "round", ylabel = "‖Aμ − b‖ / ‖b‖",
		title = "relative residual" * (fr_run.converged ? "" : "  —  DIVERGED"))
end

# ╔═╡ aabbccdd-0000-4000-8000-000000000050
md"""
| diagnostic | value | verdict |
|:--|--:|:--|
| diagonal dominance margin ``\min_i (\lvert A_{ii}\rvert - \sum_{j\neq i}\lvert A_{ij}\rvert)`` | $(@sprintf("%+.3f", fr_diag.dd)) | $(fr_diag.dd > 0 ? "dominant" : "not dominant") |
| walk-summability ``\rho(\lvert I - D^{-1}A\rvert)`` | $(@sprintf("%.3f", fr_diag.rho)) | $(fr_diag.rho < 1 ? "walk-summable" : "not walk-summable") |
| smallest eigenvalue ``\lambda_{\min}(A)`` | $(@sprintf("%+.3f", fr_diag.lam)) | $(fr_diag.lam > 0 ? "valid Gaussian" : "not a distribution") |
| GaBP | $(fr_run.iters) rounds | $(fr_run.converged ? "converged" : "diverged") |
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000051
md"""
!!! warning "Try it"
	Push ``w`` up from 0.05. The unit diagonal is beaten by the four couplings at ``w = 0.25`` and walk-summability goes one step later at ``w \approx 0.26`` — the two sufficient conditions fail together, and neither failure costs anything: the solver keeps converging, taking 45 rounds at ``w = 0.26`` and 353 at ``w = 0.29``. It breaks between ``w = 0.29`` and ``w = 0.30``, which is essentially where ``A`` stops being positive definite (``\lambda_{\min} = +0.010`` at ``w = 0.30``).

	Then turn on damping, ``P \leftarrow (1-\alpha)P_{\text{new}} + \alpha P_{\text{old}}``: it buys smoothness in the borderline regime but does **not** rescue the indefinite case — and it should not, because there is no valid Gaussian left to infer. Sharp characterisations of the convergence basin, and principled fixes outside it, remain open (see Johnson et al. 2009, Ruozzi & Tatikonda 2013).
"""

# ╔═╡ aabbccdd-0000-4000-8000-000000000052
md"""
# 5. Where this goes

We arrived at a linear solver that is local, asynchronous, communication-light, and reports a per-node uncertainty as a by-product — and whose classical counterpart (Jacobi) is literally itself with the second moment deleted. That combination is the argument of this tutorial: **probabilistic numerics at scale wants message passing, because message passing is what turns a global belief into a distributed one.**

Possible research directions:

* **Beyond symmetry.** GaBP as derived needs ``A = A^\top``. Shental et al. (§VII) embed a rectangular ``S`` into a symmetric system whose solution is the ridge/pseudo-inverse estimate ``(S^\top S + \Psi)^{-1}S^\top y`` — with ``2nk`` messages rather than ``n^2``. Fanaskov (2022) instead modifies the messages themselves for non-symmetric ``A``, relates the result to LU and block-LU factorisation, and uses GaBP as a **multigrid smoother**, where it is markedly more robust than incomplete-LU or Gauss–Seidel smoothing.
* **Beyond linear systems.** Because an interior-point method is a sequence of linear systems (Newton steps on the Hessian), swapping each solve for GaBP gives a **distributed linear-programming solver** (Bickson et al. 2008). The same substitution works anywhere a Newton step is the inner loop.
* **Better uncertainty.** Two distinct problems, and this tutorial solves neither. *First*, the means are exact on convergence but the variances are not: generalised BP / the cluster-variation method (the second algorithm in Fanaskov 2022), or the walk-sum corrections of Johnson et al., buy calibration by giving up locality. **What is the cheapest message-passing scheme with honest variances?** *Second*, and more fundamental (§4.5): the belief is a statement about the ``k``-hop sub-problem, not about the error, because the precision recursion never sees ``b``. An anytime *belief* is not an anytime *error estimate*, and nothing in the classical GaBP literature is trying to make it one. **What would a message that carried error information — rather than only information about ``A`` — even look like?** Both are probabilistic-numerics questions, not linear-algebra ones; the second is the one we would most like an answer to.
* **Applications where the graph is real.** Power-grid state estimation, sensor-network localisation, SLAM and bundle adjustment (Gaussian BP is the engine of several modern SLAM back-ends), CDMA multiuser detection — in each case the factor graph is not a metaphor for the sparsity pattern; it is the physical layout of the machine. The die of §1 is the limiting case, where the graph is *literally* the silicon; and that this is a good bargain in wall-clock, not just in rhetoric, has been measured. Ortiz et al. (2020) solved a real bundle-adjustment problem by GaBP on the 1216 cores of a single graph processor in under 40 ms, against 1450 ms for a sparse-Cholesky CPU library — the whole margin coming from an algorithm that never needs anything but nearest-neighbour exchange.

### References

* Shental, O., Bickson, D., Siegel, P. H., Wolf, J. K., & Dolev, D. (2008). *Gaussian belief propagation solver for systems of linear equations*. IEEE ISIT, 1863–1867. Extended: [arXiv:0810.1119](https://arxiv.org/abs/0810.1119).
* Bickson, D., Tock, Y., Shental, O., & Dolev, D. (2008). *Polynomial linear programming with Gaussian belief propagation*. Allerton, 895–901.
* Fanaskov, V. (2022). *Gaussian belief propagation solvers for nonsymmetric systems of linear equations*. SIAM J. Sci. Comput., 44(2), A77–A102.
* Weiss, Y., & Freeman, W. T. (2001). *Correctness of belief propagation in Gaussian graphical models of arbitrary topology*. Neural Computation, 13(10), 2173–2200.
* Malioutov, D. M., Johnson, J. K., & Willsky, A. S. (2006). *Walk-sums and belief propagation in Gaussian graphical models*. JMLR, 7, 2031–2064.
* Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic Numerics: Computation as Machine Learning*. Cambridge University Press.
* Cockayne, J., Oates, C. J., Ipsen, I. C. F., & Girolami, M. (2019). *A Bayesian conjugate gradient method*. Bayesian Analysis, 14(3), 937–1012.
* Ortiz, J., Pupilli, M., Leutenegger, S., & Davison, A. J. (2020). *Bundle adjustment on a graph processor*. CVPR, 2413–2422. [arXiv:2003.03134](https://arxiv.org/abs/2003.03134).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
"""

# ╔═╡ Cell order:
# ╠═aabbccdd-0000-4000-8000-000000000001
# ╟─aabbccdd-0000-4000-8000-000000000002
# ╟─aabbccdd-0000-4000-8000-000000000003
# ╟─aabbccdd-0000-4000-8000-000000000004
# ╟─aabbccdd-0000-4000-8000-000000000005
# ╠═aabbccdd-0000-4000-8000-000000000006
# ╟─aabbccdd-0000-4000-8000-000000000007
# ╟─aabbccdd-0000-4000-8000-000000000008
# ╟─aabbccdd-0000-4000-8000-000000000009
# ╠═aabbccdd-0000-4000-8000-000000000010
# ╟─aabbccdd-0000-4000-8000-000000000011
# ╟─aabbccdd-0000-4000-8000-000000000012
# ╟─aabbccdd-0000-4000-8000-000000000013
# ╠═aabbccdd-0000-4000-8000-000000000014
# ╠═aabbccdd-0000-4000-8000-000000000015
# ╟─aabbccdd-0000-4000-8000-000000000016
# ╠═aabbccdd-0000-4000-8000-000000000017
# ╟─aabbccdd-0000-4000-8000-000000000018
# ╟─aabbccdd-0000-4000-8000-000000000019
# ╠═aabbccdd-0000-4000-8000-000000000020
# ╟─aabbccdd-0000-4000-8000-000000000021
# ╟─aabbccdd-0000-4000-8000-000000000022
# ╟─aabbccdd-0000-4000-8000-000000000023
# ╠═aabbccdd-0000-4000-8000-000000000024
# ╟─aabbccdd-0000-4000-8000-000000000025
# ╟─aabbccdd-0000-4000-8000-000000000026
# ╟─aabbccdd-0000-4000-8000-000000000027
# ╟─aabbccdd-0000-4000-8000-000000000028
# ╟─aabbccdd-0000-4000-8000-000000000029
# ╟─aabbccdd-0000-4000-8000-000000000030
# ╟─aabbccdd-0000-4000-8000-000000000031
# ╟─aabbccdd-0000-4000-8000-000000000032
# ╟─aabbccdd-0000-4000-8000-000000000033
# ╠═aabbccdd-0000-4000-8000-000000000034
# ╟─aabbccdd-0000-4000-8000-000000000035
# ╟─aabbccdd-0000-4000-8000-000000000036
# ╠═aabbccdd-0000-4000-8000-000000000037
# ╟─aabbccdd-0000-4000-8000-000000000038
# ╟─aabbccdd-0000-4000-8000-000000000039
# ╟─aabbccdd-0000-4000-8000-000000000040
# ╟─aabbccdd-0000-4000-8000-000000000041
# ╠═aabbccdd-0000-4000-8000-000000000042
# ╟─aabbccdd-0000-4000-8000-000000000043
# ╟─aabbccdd-0000-4000-8000-000000000044
# ╟─aabbccdd-0000-4000-8000-000000000045
# ╠═aabbccdd-0000-4000-8000-000000000046
# ╟─aabbccdd-0000-4000-8000-000000000047
# ╠═aabbccdd-0000-4000-8000-000000000048
# ╟─aabbccdd-0000-4000-8000-000000000049
# ╟─aabbccdd-0000-4000-8000-000000000050
# ╟─aabbccdd-0000-4000-8000-000000000051
# ╟─aabbccdd-0000-4000-8000-000000000052
# ╟─00000000-0000-0000-0000-000000000001
