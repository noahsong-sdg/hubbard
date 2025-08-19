I asked gpt-5-high to generate an MD document with cool things I could do after I recreated the Claveau et al paper. 

---

## FUTURE experiments and analyses

This document lists possible experiments and analyses to extend the current Hubbard codebase toward SSH and beyond. Items are grouped from baseline (known) to likely novel directions. Use OBC vs PBC explicitly, specify unit cell (two-site A/B), and keep consistent conventions across modules.

### Baseline validation (known; build confidence)
- **Non-interacting SSH (1D)**:
  - Implement alternating hoppings `t1`, `t2`; confirm band gap `|t1 - t2|` and edge modes for OBC when `|t2| > |t1|`.
  - Compute/plot: bands, DOS (1D), edge-localized midgap states (OBC).
  - Parameter sweep: `t1/t2 ∈ [0.2, 5]`, `L ∈ {40, 80, 160}`, OBC and PBC.
- **Zak phase / polarization (non-interacting)**:
  - PBC: Berry phase along BZ; OBC: bulk–boundary correspondence via midgap states.
  - Cross-check gauge/mesh convergence.
- **SSH–Hubbard at mean-field (MF)**:
  - Use 2×2 k-space MF with SSH off-diagonal `f(k)=t1 + t2 e^{-ik}`; diagonal `U ⟨n̄A/B⟩`.
  - Phases vs `(δ = (t1−t2)/(t1+t2), U/t̄)`; compare PM/FM/AFM seeds.

### Interacting topology beyond MF (novelty likely)
- **Ground-state phase diagram (SSH–U–V)**:
  - Model: 1D chain with alternating bonds, on-site `U` and nearest-neighbor `V`.
  - Methods: DMRG (OBC, finite), iDMRG (PBC/two-site unit cell) for thermodynamic limit.
  - Order parameters: BOW (bond order), CDW (density), SDW (spin), many-body polarization (Resta), entanglement spectrum/degeneracy.
  - Outputs: phase diagram in `(δ, U/t̄, V/t̄)`, critical lines, central charge from entanglement scaling.
  - Parameters: `L=96–256` (OBC), bond dimension `χ=200–1000`, truncation `1e−8`, sweeps `10–30`.
- **Edge-state robustness with interactions**:
  - OBC: edge magnetization, midgap many-body states vs `(U, V)`.
  - Measure localization length of edge excitations; finite-size scaling vs `L`.

### Disorder and topology (interacting topological Anderson physics)
- **Diagonal disorder `W` on sites and/or bond disorder on `t1,t2`**:
  - Phase diagram in `(δ, U, V, W)`; identify disorder-induced topology and many-body gaps.
  - Observables: polarization distribution, entanglement spectrum statistics, inverse participation ratio of edge modes (many-body analog via local spectral function).
  - Average over `N_realizations = 50–200`; sizes `L=64–160`.

### Floquet and driven phases (heating and prethermal regimes)
- **Periodic modulation of dimerization**: `t1(t)=t̄(1+δ0+δ1 cosΩt)`, `t2(t)=t̄(1−δ0−δ1 cosΩt)`.
  - Non-interacting: compute Floquet quasi-bands and winding numbers; edge modes at π/T.
  - Interacting: TEBD/tDMRG with high-frequency drive; prethermal window length vs `U, Ω`.
  - Diagnostics: energy absorption rate, stroboscopic polarization, entanglement growth.

### Quench dynamics (dynamical topology)
- **Quench δ: trivial → topological and vice versa** (with/without `U,V`).
  - Track: Loschmidt echo/zeros, dynamical polarization, light-cone spread of correlations, edge-mode emergence time.
  - TEBD parameters: time step `dt=0.05–0.1/t̄`, total time `T=50–200/t̄`, bond dim adaptive up to `χ≈2000`.

### Finite temperature (robustness in realistic settings)
- **METTS or purification** for SSH–U–V at `T>0`.
  - Quantify crossover temperature for edge occupancy and polarization.
  - Thermodynamics: specific heat, susceptibility, gap closing/opening with `T`.

### Extended models and crossovers
- **Longer-range terms**: next-nearest-neighbor hopping, staggered on-site potentials, spin–orbit (BDI → AIII crossovers), pairing (BDG) for topological superconductivity.
- **Coupling to environments**: Lindblad dephasing; steady-state edge coherence times.

### Measurement links (make theory testable)
- **Cold atoms**:
  - Superlattice realizations; predict: edge densities (QGM images), Thouless pumping trajectories, Bragg spectroscopy of correlations.
- **Photonics/electrical**:
  - Midgap transmission peaks, disorder-robust edge transport; driven protocols for Floquet edge modes.

### Concrete task list and outputs
- **Implementational tasks**:
  - Add SSH mode: `t1,t2`, OBC/PBC toggle; 1D bands/DOS utilities; keep legacy 2D paths intact.
  - Many-body invariants: Resta polarization (PBC), entanglement spectrum (OBC/PBC), BOW/CDW/SDW.
- **Data products**:
  - Band/DOS plots (1D), edge-localization profiles, phase diagrams `(δ,U,V[,W])`, scaling collapses, central charge fits, quench dynamics traces, finite-T curves.
- **Validation checkpoints**:
  - Non-interacting limits reproduce analytic results; MF recovers expected trends; DMRG benchmarks at `U=0` and small `U` vs MF.

### Suggested parameter grids (starting points)
- **Baseline**: `δ ∈ {−0.6..0.6}` (11 steps), `U/t̄ ∈ {0, 1, 2, 4, 6}`, `V/t̄ ∈ {0, 1, 2}`.
- **Disorder**: `W/t̄ ∈ {0, 0.5, 1.0, 2.0}`, realizations: 100.
- **Floquet**: `Ω/t̄ ∈ {6, 10, 20}`, `δ1 ∈ {0.1, 0.3, 0.5}`.
- **Finite-T**: `T/t̄ ∈ {0.01..0.5}` logarithmic.

### Milestones
- **M1**: Non-interacting SSH with OBC/PBC, bands/DOS, edge states validated.
- **M2**: MF SSH–Hubbard phase trends; k-space 1D pipeline complete.
- **M3**: DMRG phase diagram with many-body polarization and entanglement diagnostics.
- **M4**: One advanced axis (disorder, Floquet, dynamics, or finite-T) with scaling quality figures.

### Notes on reproducibility
- Fix random seeds; store Hamiltonian definitions and parameter JSON/YAML alongside plots.
- Automate sweeps; log convergence metrics (energy variance, discarded weight, entanglement entropy) and wall-clock.

---

If you choose one “advanced axis” (disorder, Floquet, dynamics, or finite-T) and run it to completion with clean scaling, that is a plausible path to a publishable result.

