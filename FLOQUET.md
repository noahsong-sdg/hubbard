## Studying Floquet effects in the SSH model

This document lays out a practical plan to implement, validate, and extend Floquet physics in the SSH (and SSH–Hubbard) model using this codebase.

### Scope
- **Targets**: non-interacting Floquet SSH, then interacting (SSH–Hubbard) under periodic drive.
- **Deliverables**: quasienergy bands (0 and π/T gaps), edge states (OBC), topological invariants, heating/prethermal behavior, and parameter-phase diagrams.

### Model and drives
- **Static SSH**: two-site unit cell (A/B), alternating bonds `t1`, `t2`.
- **Periodic driving options**:
  - **Bond modulation (recommended)**: `t1(t)=t̄(1+δ0+δ1 cos Ω t)`, `t2(t)=t̄(1−δ0−δ1 cos Ω t)`.
  - **Staggered on-site potential**: `Δ(t) σz` (breaks chiral symmetry; use with care if targeting BDI).
  - **Step (two-step) drive**: alternate `(t1,t2)` values per time slice to realize analytic invariants more easily.

### Core formalism
- **Floquet operator**: `U(T) = Texp(−i ∫_0^T H(t) dt)`; diagonalize `U(T)` to get quasienergies `ε` via phases `e^{−i ε T}`.
- **Bloch approach (U=0, PBC)**: for each k, form `H_k(t)` (2×2) and compute `U_k(T)` by time discretization: `U_k(T) ≈ ∏_n exp(−i H_k(t_n) Δt)`.
- **Real-space approach (U=0, OBC)**: build single-particle `H(t)` on L sites and compute `U(T)`; diagnose edge modes by eigenstate localization.
- **Interacting**: time evolution of many-body state by Trotterized TEBD/tDMRG (or Krylov for small L).

### Experiments to run
- **E1 — Quasienergy bands and edge modes (U=0)**
  - Compute `U_k(T)` on a dense 1D k-grid; plot quasienergy bands in reduced Floquet zone `ε ∈ (−π/T, π/T]`.
  - OBC: diagonalize `U(T)`; identify states at `ε≈0` and `ε≈π/T` localized at edges; plot inverse participation ratio vs site.
  - Sweep `(Ω/t̄, δ1, δ0)`; map presence/absence of 0/π edge modes.
- **E2 — Topological invariants (U=0)**
  - Compute effective Hamiltonian `H_F(k) = (i/T) Log[U_k(T)]` with two branch cuts (0-gap and π-gap) to obtain windings `W_0`, `W_π`.
  - Validate that `(W_0, W_π)` predict counts of 0/π edge modes in OBC.
- **E3 — Heating and prethermal plateaus (U>0)**
  - TEBD/tDMRG at half-filling; drive bonds as above; track energy density and entanglement entropy vs time for `Ω ≫ bandwidth` and `Ω ~ bandwidth`.
  - Extract heating rates and prethermal window length vs `(U, Ω, δ1)`.
- **E4 — Interacting edge signatures (U>0)**
  - OBC: initialize localized edge excitation; evolve stroboscopically and measure survival probability/localization length vs time and parameters.
  - Compare with non-interacting edge-state robustness.
- **E5 — Phase diagrams**
  - Non-interacting: chart regions with `(W_0, W_π)` over `(Ω, δ1, δ0)`.
  - Interacting (small L): Krylov/ED to obtain stroboscopic spectra and level statistics; TEBD for larger L to map heating/no-heating regimes.

### Software stack
- **Julia-first (recommended for this repo)**:
  - Linear algebra: `LinearAlgebra`, `SparseArrays` (already in use).
  - Time evolution (single-particle): direct matrix exponentials per slice via `expm` or diagonalization of 2×2 `H_k(t)`; for real-space, Krylov time-stepping (e.g., `Expokit.jl` or custom Lanczos) or small-Δt product formula.
  - Tensor networks (interacting): `ITensors.jl` for DMRG/TEBD (time evolution, bond dimension control, observables).
  - Optional ODE: `DifferentialEquations.jl` if using state-vector ODE integration for small systems.
- **Python alternative (optional)**:
  - `QuTiP` for time-dependent Hamiltonians and Floquet analysis; `NumPy`/`SciPy` for numerics.

### Procedures (U=0)
- **P1 — Bloch (2×2) pipeline**
  - Implement `H_k(t) = [0  f(k,t); f*(k,t)  0]` with `f(k,t) = t1(t) + t2(t) e^{−ik}`.
  - Choose `N_t` time slices with `Δt = T/N_t`; compute `U_k(T)` by ordered product of `exp(−i H_k(t_n) Δt)`.
  - Diagonalize `U_k(T)` → `ε_n(k)`; unwrap to `(−π/T, π/T]`.
  - Compute `H_F(k)` using principal log (0-gap) and shifted branch (π-gap); for each branch, compute winding of Bloch vector to get `W_0`, `W_π`.
- **P2 — Real-space OBC**
  - Build time-dependent single-particle `H(t)` on length `L` with alternating bonds; OBC.
  - Compute `U(T)` via time-slice product; diagonalize; plot eigenphases and edge localization.

### Procedures (U>0)
- **P3 — TEBD (interacting)**
  - Hamiltonian split into bonds; apply second-order (or fourth-order) Trotter with time-dependent bond strengths.
  - At stroboscopic times `nT`, record: energy per site, entanglement entropy, local densities/bond orders; assess heating and prethermal plateaus.
  - Parameters: `dt = 0.02–0.05 / t̄`, `χ up to 1000–2000`, truncate `1e−8`.
- **P4 — Small-L Krylov/ED**
  - For `L ≤ 14` sites, evolve many-body state with Krylov integrator; compute stroboscopic spectrum `U(T)` in full Hilbert space when feasible.

### Parameter grids (starting points)
- Frequencies: `Ω/t̄ ∈ {6, 10, 20}` (high-frequency), `{2, 3, 4}` (near-resonant).
- Amplitudes: `δ1 ∈ {0.1, 0.3, 0.5}`, offsets `δ0 ∈ {0.0, 0.2}`.
- Sizes: `L ∈ {100, 200}` (single-particle OBC), `L ∈ {50, 100}` (TEBD), fillings: half-filling.
- Interactions: `U/t̄ ∈ {0, 1, 2, 4}`, optionally nearest-neighbor `V/t̄ ∈ {0, 1}`.
- Time discretization: ensure `N_t ≥ 400` per period for smooth drives (check convergence).

### Validation and reproduction (@Web)
- **Baseline Floquet SSH with edge states**:
  - Reproduce quasienergy bands and 0/π edge modes under bond modulation as in standard Floquet-SSH literature (e.g., step/two-step protocols).
  - Reference: Asbóth et al., “A Short Course on Topological Insulators” (Floquet chapter) — [@Web: textbook overview](https://topocondmat.org/w4_floquet/).
- **PT-symmetric Floquet SSH** (optional non-Hermitian check):
  - Analyze quasi-energy spectrum and stability regions under gain/loss modulation; validate phase boundaries.
  - Reference: “PT Symmetric Floquet Topological Phase in SSH Model” — [@Web: arXiv:1806.06722](https://arxiv.org/abs/1806.06722).
- **Driven extended SSH with high Chern numbers** (2D pump mapping / extended couplings):
  - Benchmark emergence of multiple topological phases vs drive amplitude/frequency.
  - Reference: “Floquet Topological Phases with High Chern Numbers in a Periodically Driven Extended SSH Model” — [@Web: arXiv:2201.10884](https://arxiv.org/abs/2201.10884).

### Outputs and figures
- Quasienergy bands `ε(k)` with marked 0/π gaps.
- OBC eigenphase spectra with edge localization heatmaps.
- Invariants `(W_0, W_π)` across parameter grids; confusion matrix vs edge counts.
- Heating curves (energy, entropy) vs time for interacting drives; prethermal lifetime vs `Ω`.

### Implementation notes for this repo
- Add an SSH mode for 1D and OBC/PBC if not present.
- Create utilities:
  - `floquet_Uk(k, params)::Matrix{ComplexF64}` building `U_k(T)` by time-slicing.
  - `floquet_invariants(params)::(W0, Wpi)` using branch-cut logs.
  - `floquet_U_realspace(params)::Matrix{ComplexF64}` for OBC edge analysis.
- Keep the 2D codepaths intact; guard new 1D/Floquet routines behind clear flags.

### Repro checklist
- Convergence in `Δt`, `k`-grid, and (for TEBD) `χ`.
- Branch-cut choice verified for 0/π invariants.
- OBC/PBC consistency: invariants predict edge counts.
- Seeds and metadata saved with figures for reproducibility.

---

This plan gets you from non-interacting Floquet SSH validation to interacting prethermal/heating studies with publishable-quality figures, while staying close to your current Julia stack. Optionally, mirror single-particle results in Python/QuTiP for cross-checking if desired.
