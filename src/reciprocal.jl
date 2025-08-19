# Functions that compute bands and DOS for various occupation levels
# This file is included directly into HubbardModel module

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf

# ----------------------------------------------------------------------------- #
# --- High-Symmetry Points and Paths --- 


# ----------------------------------------------------------------------------- #
# --- Helper Functions --- 

function calculate_dispersion(params::HubbardParams, k_points)
    # For non-interacting case (U=0), we can use the tight-binding dispersion
    t = params.t
    L = params.L
    
    # Assuming 2D square lattice with L = L_x × L_y
    L_x = Int(sqrt(L))
    L_y = L ÷ L_x
    
    # Calculate dispersion along the path
    energies = []
    for (kx, ky) in k_points
        # Tight-binding dispersion in 2D: -2t(cos(kx) + cos(ky))
        ε_k = -2*t*(cos(kx) + cos(ky))
        push!(energies, ε_k)
    end
    
    return energies
end
function generate_k_path(; 
    points::Vector{Any} = [], 
    labels::Vector{String} = String[], 
    num_points_per_segment::Int = 50, 
    supercell::Bool = false,
    ssh_mode::Bool = false
)
    """
    generate_k_path(; points, labels, num_points_per_segment=50, supercell=false, ssh_mode=false)

    - If `points` is empty, picks the default path:
        • ssh_mode=true: Γ–X–Γ (1D SSH chain)
        • supercell=false: Γ–X–M–Γ (2D square lattice)
        • supercell=true:  Γ_b–M_b–Γ_b–X_b–Γ_b (2D supercell)
    - If you pass `points`, you *must* also pass `labels` of the same length.
    - Returns (k_path, k_dist, tick_positions, tick_labels).
    """

    # 1) Defaults if user didn’t supply points & labels
    if isempty(points)
        if supercell
            points = [[0.0,0.0], [π,π], [0.0,0.0], [π,0.0], [0.0,0.0]]
            labels = ["Γ_b","M_b","Γ_b","X_b","Γ_b"]
        else
            points = [[0.0,0.0], [π,0.0], [π,π], [0.0,0.0]]
            labels = ["Γ","X","M","Γ"]
        end
    else
        # 2) If user gave points, they must also give matching labels
        if length(labels) != length(points)
            error("When you pass `points`, you must also pass a `labels` array of the same length.")
        end
    end

    # 3) Build the discrete path and distances
    k_path = Tuple{Float64,Float64}[]
    k_dist = Float64[0.0]
    tick_positions = Float64[0.0]

    current_dist = 0.0
    n = num_points_per_segment

    for i in 1:length(points)-1
        p1 = points[i]
        p2 = points[i+1]
        Δ = p2 .- p1

        # for the last segment include endpoint exactly once
        steps = (i == length(points)-1) ? 0 : n-1
        for s in 0:n
            t = s/n
            kx, ky = p1 .+ t .* Δ
            if (i==1 && s==0)
                push!(k_path,(kx,ky))
            elseif s>0
                push!(k_path,(kx,ky))
                prev = k_path[end-1]
                d = hypot(kx-prev[1], ky-prev[2])
                current_dist += d
                push!(k_dist, current_dist)
            end
        end

        push!(tick_positions, current_dist)
    end

    # 4) Sanity-check
    if length(k_path) != length(k_dist)
        # recompute distances if something went awry
        k_dist = zeros(length(k_path))
        for j in 2:length(k_path)
            k_dist[j] = k_dist[j-1] + hypot(
                k_path[j][1]-k_path[j-1][1],
                k_path[j][2]-k_path[j-1][2]
            )
        end
    end

    return k_path, k_dist, tick_positions, labels
end

function gaussian_dos(energies, ω_grid, σ::Float64)
    """
    gaussian_dos(energies, ω_grid, σ)

    Calculates the Density of States (DOS) using Gaussian broadening.
    Normalized to give DOS per single unit cell per spin.
    """
    dos = zeros(length(ω_grid))
    if isempty(energies)
        return dos
    end
    norm_factor = 1.0 / (σ * sqrt(2π))
    for i in eachindex(ω_grid)
        ω = ω_grid[i]
        dos[i] = sum(norm_factor * exp.(-0.5 .* ((ω .- energies) ./ σ).^2))
    end
    # length(energies) = Nk_dos * Nk_dos * num_bands (which is 2 for the double cell)
    # Normalize by total k-points (Nk_dos*Nk_dos = length(energies)/2) and number of sites (2)
    total_k_points_times_sites = length(energies) 
    if total_k_points_times_sites > 0
         dos ./= total_k_points_times_sites # Normalize to per site (single cell), per spin
    end
    return dos
end

# ---------------------------------------------------
# --- Core Calculation Functions --- ------------------
# --------------------------------------------------

function calculate_fermi_level(params::MFParams, nup, ndown; Nk_grid=50)
    """
    calculate_fermi_level(params::MFParams, nup, ndown; Nk_grid=50)

    Calculates the Fermi level (chemical potential at T=0) required to achieve
    the target occupations nup and ndown.

    Args:
        params: Mean-field parameters.
        nup: Target average occupation per site for spin up.
        ndown: Target average occupation per site for spin down.
        Nk_grid: The size of the k-point grid (Nk_grid x Nk_grid) used for calculation.

    Returns:
        E_F: The calculated Fermi level.
    """
    println("Calculating Fermi level on $Nk_grid x $Nk_grid grid...")
    kx_grid = LinRange(0, 2π * (1 - 1/Nk_grid), Nk_grid)
    ky_grid = LinRange(0, 2π * (1 - 1/Nk_grid), Nk_grid)
    grid = [(kx, ky) for kx in kx_grid, ky in ky_grid]

    # Calculate all eigenvalues
    println("  Calculating eigenvalues for Fermi level...")
    # Note: Pass n_opposite_spin to mean_field_hamiltonian
    all_energies_up = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndown, params))) for (kx, ky) in grid]...)
    all_energies_dn = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, params))) for (kx, ky) in grid]...)
    all_energies = vcat(all_energies_up, all_energies_dn)

    if isempty(all_energies)
        println("  Warning: No eigenvalues found for Fermi level calculation.")
        return 0.0 # Default Fermi level if no states
    end

    # Sort energies
    sort!(all_energies)
    N_states = length(all_energies)

    # Determine number of electrons based on filling per site and number of k-points.
    # Assumes Nk_grid*Nk_grid is the number of k-points in the BZ being sampled.
    # The number of electrons to accommodate in the calculated states.
    num_k_points = Nk_grid * Nk_grid
    # This definition assumes the total number of electrons corresponds directly to the number of states to fill,
    # based on the total filling (nup + ndown) relative to the number of k-points sampled.
    # This interpretation might need adjustment depending on the exact structure
    # of the mean-field Hamiltonian (e.g., number of bands, unit cell doubling).
    idx_fermi = round(Int, mean(nup .+ ndown) * num_k_points)

    # Clamp index to valid range
    idx_fermi = max(1, min(N_states, idx_fermi))

    # Estimate Fermi level (e.g., midpoint between occupied and unoccupied)
    E_F = 0.0
    if N_states == 0
        E_F = 0.0
    elseif idx_fermi == N_states
        E_F = all_energies[N_states] # System is full
    elseif idx_fermi == 0
        E_F = all_energies[1] # System is empty (use lowest state energy)
    else
        # Midpoint between highest occupied and lowest unoccupied state
        E_F = (all_energies[idx_fermi] + all_energies[idx_fermi + 1]) / 2.0
    end

    println("  Calculated Fermi level E_F = $E_F for nup=$nup, ndown=$ndown")
    return E_F
end

function calculate_bands(params::MFParams, nup, ndown, k_path; E_F::Float64 = 0.0)
    """
    calculate_bands(params::MFParams, nup, ndown, k_path; E_F=0.0)

    Calculates band structure eigenvalues along a given k-path.
    Optionally shifts eigenvalues so the Fermi level E_F is at zero.
    
    Handles both 2D (standard) and 1D (SSH) k-paths automatically.

    Returns eigenvalues for spin up (em_up) and spin down (em_dn), shifted if E_F is provided.
    """
    println("Calculating band structure...")
    
    if params.ssh_mode
        # SSH mode: 1D k-path, use SSH Hamiltonian
        evals_up = [sort(eigvals(Hermitian(mean_field_hamiltonian_ssh(kx, ndown, params)))) for kx in k_path]
        evals_dn = [sort(eigvals(Hermitian(mean_field_hamiltonian_ssh(kx, nup, params)))) for kx in k_path]
    else
        # Standard mode: 2D k-path, use standard Hamiltonian
        evals_up = [sort(eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndown, params)))) for (kx, ky) in k_path]
        evals_dn = [sort(eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, params)))) for (kx, ky) in k_path]
    end

    em_up = evals_up
    em_dn = evals_dn

    # Shift energies by -E_F
    shifted_em_up = [[e - E_F for e in evals] for evals in em_up] # Equivalent list comprehension
    shifted_em_dn = [[e - E_F for e in evals] for evals in em_dn] # Equivalent list comprehension

    println("Band structure calculation complete. Energies shifted by E_F = $E_F.")
    return shifted_em_up, shifted_em_dn
end

"""
calculate_zak_phase_ssh(params::MFParams, nup, ndown; spin=:up, Nk::Int=401)

Computes the Zak phase (1D Berry phase) for the occupied band of the SSH model
over k ∈ [0, 2π). Requires `params.ssh_mode == true`.

Args:
    params: Mean-field parameters (SSH mode must be enabled)
    nup:    Spin-up densities on sublattices [A, B]
    ndown:  Spin-down densities on sublattices [A, B]
    spin:   :up or :down – which spin channel Hamiltonian to use
    Nk:     Number of k-points along the loop (use an odd number to include k=π)

Returns:
    Float64: Zak phase in [0, 2π)
"""
function calculate_zak_phase_ssh(params::MFParams, nup, ndown; spin::Symbol=:up, Nk::Int=401)
    params.ssh_mode || error("calculate_zak_phase_ssh requires params.ssh_mode = true")
    Nk < 3 && error("Nk must be ≥ 3")

    ks = range(0.0, 2π, length=Nk)

    # Collect occupied-band eigenvectors along k
    occupied_vectors = Vector{Vector{ComplexF64}}(undef, Nk)
    for (i, kx) in enumerate(ks)
        if spin === :up
            H = mean_field_hamiltonian_ssh(kx, ndown, params)
        elseif spin === :down
            H = mean_field_hamiltonian_ssh(kx, nup, params)
        else
            error("spin must be :up or :down")
        end

        vals, vecs = eigen(Hermitian(H))
        occupied_vectors[i] = vecs[:, 1] # lower band
    end

    # Discrete Wilson loop: product of overlaps of neighboring k-points (phase only)
    prod_phase = 1.0 + 0im
    for i in 1:Nk-1
        olap = dot(occupied_vectors[i], occupied_vectors[i+1])
        if olap == 0
            # Rare gauge singularity; nudge by tiny imaginary part
            olap += 1e-16im
        end
        prod_phase *= olap / abs(olap)
    end
    # Close the loop: last to first
    olap = dot(occupied_vectors[Nk], occupied_vectors[1])
    if olap == 0
        olap += 1e-16im
    end
    prod_phase *= olap / abs(olap)

    zak = mod2pi(angle(prod_phase))
    return zak
end

"""
classify_ssh_topology(params::MFParams, nup, ndown; Nk::Int=401, tol::Float64=0.2)

Classifies the SSH phase (trivial vs topological) using the Zak phase of the
occupied band. Returns :trivial, :topological, or :ambiguous.
"""
function classify_ssh_topology(params::MFParams, nup, ndown; Nk::Int=401, tol::Float64=0.2)
    zak = calculate_zak_phase_ssh(params, nup, ndown; spin=:up, Nk=Nk)
    # Map Zak to nearest {0, π}
    if min(abs(zak - 0.0), abs(2π - zak)) < tol
        return :trivial, zak
    elseif min(abs(zak - π), abs(zak - (2π - π))) < tol
        return :topological, zak
    else
        return :ambiguous, zak
    end
end

# --------------------------------------------------------
# --- Floquet (non-interacting, Bloch 2×2) ---------------
# --------------------------------------------------------

"""
ssh_bloch_hamiltonian(kx::Float64, t1::Real, t2::Real)

Returns the 2×2 non-interacting SSH Bloch Hamiltonian at momentum kx:
    H(k) = [ 0      f(k) ]
           [ f*(k)  0    ]
with f(k) = t1 + t2 e^{-ik}.
"""
function ssh_bloch_hamiltonian(kx::Float64, t1::Real, t2::Real)
    f = t1 + t2 * exp(-1im * kx)
    return ComplexF64[ 0.0               f
                       conj(f)           0.0 ]
end

"""
floquet_Uk_bondmod(kx; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400)

Computes the single-period Floquet operator U_k(T) for the SSH model with a
bond-modulation drive:
    t1(t) = tbar * (1 + δ0 + δ1 cos Ω t)
    t2(t) = tbar * (1 - δ0 - δ1 cos Ω t)
by a time-ordered product of short evolutions over Nt slices.
"""
function floquet_Uk_bondmod(kx; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400)
    T = 2π / Ω
    Δt = T / Nt
    U = Matrix{ComplexF64}(I, 2, 2)
    # Left-multiply in chronological order: U ← exp(-i H(t_n) Δt) U
    for n in 1:Nt
        t = (n - 0.5) * Δt
        t1 = tbar * (1 + δ0 + δ1 * cos(Ω * t))
        t2 = tbar * (1 - δ0 - δ1 * cos(Ω * t))
        H = ssh_bloch_hamiltonian(kx, t1, t2)
        vals, vecs = eigen(Hermitian(H))
        Ustep = vecs * Diagonal(exp.(-1im .* vals .* Δt)) * vecs'
        U = Ustep * U
    end
    return U
end

"""
floquet_quasienergies(kx; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400)

Returns quasienergies ε ∈ (−π/T, π/T] for the driven SSH model at momentum kx
using floquet_Uk_bondmod.
"""
function floquet_quasienergies(kx; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400)
    T = 2π / Ω
    U = floquet_Uk_bondmod(kx; Ω=Ω, δ0=δ0, δ1=δ1, tbar=tbar, Nt=Nt)
    λ = eigvals(U)  # on unit circle: λ = e^{-i ε T}
    phases = angle.(λ)                    # in (−π, π]
    ε = (-1.0 ./ T) .* phases             # map to quasienergies
    return sort(ε)
end

"""
calculate_floquet_bands(; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400, Nk::Int=201)

Computes quasienergy bands over k ∈ [0, 2π] for the bond-modulated SSH drive.

Returns (k_path, k_dist, eps_bands) where eps_bands is a Vector of length Nk
with each entry a sorted Vector of the two quasienergies at that k.
"""
function calculate_floquet_bands(; Ω::Real, δ0::Real=0.0, δ1::Real=0.3, tbar::Real=1.0, Nt::Int=400, Nk::Int=201)
    ks = range(0.0, 2π, length=Nk)
    k_path = collect(ks)
    k_dist = collect(0:Nk-1)
    eps = [floquet_quasienergies(k; Ω=Ω, δ0=δ0, δ1=δ1, tbar=tbar, Nt=Nt) for k in ks]
    return k_path, k_dist, eps
end

"""
plot_floquet_bands(kdist, eps_bands, title_str::String; tick_positions=[0, (length(kdist)-1)÷2, length(kdist)-1],
                   tick_labels=["Γ", "X", "Γ"], download::Bool=true, Ω::Real)

Plots quasienergy bands ε(k) in the reduced Floquet zone (−π/T, π/T], with
tick labels Γ–X–Γ by default.
"""
function plot_floquet_bands(kdist, eps_bands, title_str::String; tick_positions=[0, (length(kdist)-1)÷2, length(kdist)-1],
                            tick_labels=["Γ", "X", "Γ"], download::Bool=true, Ω::Real)
    T = 2π / Ω
    p = plot(title=title_str, xlabel="k-path", ylabel="Quasienergy ε")
    if isempty(eps_bands)
        return p
    end
    num_k = length(eps_bands)
    # two bands
    band1 = [eps_bands[i][1] for i in 1:num_k]
    band2 = [eps_bands[i][2] for i in 1:num_k]
    plot!(p, kdist, band1; label="ε₁", color=:blue)
    plot!(p, kdist, band2; label="ε₂", color=:red, linestyle=:dash)
    # Floquet zone guides
    hline!([-π/T, 0.0, π/T]; color=:gray, linestyle=:dot, label="")
    xticks!(p, tick_positions, tick_labels)
    display(p)
    download && savefig(p, "floquet_bands.png")
    return p
end

"""
build_ssh_obc_hamiltonian(L::Int, params::MFParams; include_interactions::Bool=false,
                          nup::Vector{Float64}=[0.5, 0.5], ndown::Vector{Float64}=[0.5, 0.5])

Constructs the single-particle SSH Hamiltonian for a finite chain with open
boundary conditions (OBC). Site ordering: (A1, B1, A2, B2, ..., AL, BL).

Args:
    L:      Number of unit cells (total sites = 2L)
    params: MFParams carrying t1, t2, U (ssh_mode not required here)
    include_interactions: If true, adds mean-field diagonal shifts
                          diag_A = U * n̄_A, diag_B = U * n̄_B
    nup, ndown: Spin densities per sublattice used for mean-field shifts

Returns:
    Matrix{ComplexF64}: (2L)×(2L) Hermitian matrix.
"""
function build_ssh_obc_hamiltonian(L::Int, params::MFParams; include_interactions::Bool=false,
                                   nup::Vector{Float64}=[0.5, 0.5], ndown::Vector{Float64}=[0.5, 0.5])
    L < 1 && error("L must be ≥ 1")
    t1 = params.t1
    t2 = params.t2
    H = zeros(ComplexF64, 2L, 2L)

    # Optional diagonal mean-field shifts (equal shift preserves chiral symmetry)
    if include_interactions
        # For a single-spin channel picture, use opposite-spin densities per sublattice
        # Here we apply the same shift to both spins, purely as a diagnostic option.
        diag_A = params.U * ndown[1]
        diag_B = params.U * ndown[2]
        for i in 1:L
            Ai = 2i - 1
            Bi = 2i
            H[Ai, Ai] += diag_A
            H[Bi, Bi] += diag_B
        end
    end

    # Intra-cell bonds (A_i ↔ B_i) with t1
    for i in 1:L
        Ai = 2i - 1
        Bi = 2i
        H[Ai, Bi] += t1
        H[Bi, Ai] += t1
    end

    # Inter-cell bonds (B_i ↔ A_{i+1}) with t2
    for i in 1:(L-1)
        Bi = 2i
        Aip1 = 2(i+1) - 1
        H[Bi, Aip1] += t2
        H[Aip1, Bi] += t2
    end

    return H
end

"""
analyze_edge_states(H::AbstractMatrix; n_edge_cells::Int=2, num_return::Int=4)

Returns a concise summary of low-energy states and their edge localization.

Args:
    H:              (2L)×(2L) Hermitian SSH OBC Hamiltonian
    n_edge_cells:   Number of unit cells to define "edge" region on each side
    num_return:     Number of lowest-|E| states to summarize

Returns:
    Vector of NamedTuples: (E, left_weight, right_weight)
"""
function analyze_edge_states(H::AbstractMatrix; n_edge_cells::Int=2, num_return::Int=4)
    vals, vecs = eigen(Hermitian(H))
    # Sort by |E|
    idx = sortperm(abs.(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]

    L = size(H, 1) ÷ 2
    n_edge_sites = 2 * n_edge_cells # two sites per cell
    left_sites = 1:n_edge_sites
    right_sites = (2L - n_edge_sites + 1):(2L)

    summary = NamedTuple[]
    for j in 1:min(num_return, length(vals))
        v = vecs[:, j]
        w_left = sum(abs2, @view v[left_sites])
        w_right = sum(abs2, @view v[right_sites])
        push!(summary, (E=vals[j], left_weight=w_left, right_weight=w_right))
    end
    return summary
end

"""
plot_obc_spectrum(H::AbstractMatrix; highlight_threshold::Float64=1e-3,
                  title_str::String="SSH OBC spectrum", download::Bool=true)

Plots the single-particle spectrum (eigenvalues) for an OBC SSH Hamiltonian.
States with |E| < highlight_threshold are highlighted.
"""
function plot_obc_spectrum(H::AbstractMatrix; highlight_threshold::Float64=1e-3,
                           title_str::String="SSH OBC spectrum", download::Bool=true)
    vals = eigvals(Hermitian(H))
    idx = sortperm(vals)
    vals = vals[idx]
    N = length(vals)

    colors = [abs(vals[i]) < highlight_threshold ? :red : :steelblue for i in 1:N]
    p = scatter(1:N, vals; color=colors, markersize=4,
                xlabel="state index", ylabel="E", title=title_str, legend=false)
    hline!([0.0]; color=:black, linestyle=:dash)
    display(p)
    download && savefig(p, "ssh_obc_spectrum.png")
    return p
end

"""
plot_edge_state_profiles(H::AbstractMatrix; num_states::Int=2, n_edge_cells::Int=2,
                         title_str::String="SSH edge-state profiles", download::Bool=true)

Plots |ψ(j)|^2 over sites for the lowest-|E| eigenstates of an OBC SSH Hamiltonian.
"""
function plot_edge_state_profiles(H::AbstractMatrix; num_states::Int=2, n_edge_cells::Int=2,
                                  title_str::String="SSH edge-state profiles", download::Bool=true)
    vals, vecs = eigen(Hermitian(H))
    order = sortperm(abs.(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    num = min(num_states, size(vecs, 2))

    L = size(H, 1) ÷ 2
    sites = 1:(2L)
    p = plot(title=title_str, xlabel="site index", ylabel="|ψ(j)|^2")
    for j in 1:num
        prob = abs2.(vecs[:, j])
        plot!(p, sites, prob; label=@sprintf("E=%.3e", vals[j]))
    end
    # mark edge regions
    vline!(p, [2*n_edge_cells]; color=:gray, linestyle=:dot, label="")
    vline!(p, [2L - 2*n_edge_cells + 1]; color=:gray, linestyle=:dot, label="")
    display(p)
    download && savefig(p, "ssh_edge_profiles.png")
    return p
end

"""
plot_fk_winding(t1::Real, t2::Real; Nk::Int=400, title_str::String="SSH f(k) winding",
                download::Bool=true)

Plots the path of f(k) = t1 + t2 e^{-ik} in the complex plane over k ∈ [0, 2π].
The loop encircles the origin iff t2 > t1 (topological phase).
"""
function plot_fk_winding(t1::Real, t2::Real; Nk::Int=400, title_str::String="SSH f(k) winding",
                         download::Bool=true)
    ks = range(0.0, 2π; length=Nk)
    fk = [t1 + t2 * exp(-1im * k) for k in ks]
    xs = real.(fk)
    ys = imag.(fk)
    p = plot(xs, ys; xlabel="Re f(k)", ylabel="Im f(k)", title=title_str,
             aspect_ratio=:equal, legend=false)
    scatter!(p, [0.0], [0.0]; markershape=:cross, color=:black)
    display(p)
    download && savefig(p, "ssh_fk_winding.png")
    return p
end

function calculate_dos(params::MFParams, nup, ndown;
                       E_F::Float64 = 0.0, # Add E_F argument
                       Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)
    """
    calculate_dos(params::MFParams, nup, ndown; E_F=0.0,
                  Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)

    Calculates the Density of States given occupation numbers and standard parameter set (DOS).
    Optionally shifts the energy axis so the Fermi level E_F is at zero.

    Returns the energy grid (ω_grid) and DOS for spin up/down (dos_up, dos_dn).
    The energy grid is centered around the shifted energies.
    """
    println("\nCalculating DOS on $Nk_dos x $Nk_dos grid...")
    # generate the Brillouin zone grid
    kx_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    ky_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    grid = [(kx, ky) for kx in kx_grid, ky in ky_grid]

    # Calculate raw eigenvalues for DOS
    println("Calculating eigenvalues...")
    # Note: Pass n_opposite_spin to mean_field_hamiltonian
    raw_dos_up_func(p, ndn) = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndn, p))) for (kx, ky) in grid]...)
    raw_dos_dn_func(p, nup) = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, p))) for (kx, ky) in grid]...)
    raw_dos_up = raw_dos_up_func(params, ndown)
    raw_dos_dn = raw_dos_dn_func(params, nup)

    # Shift raw energies by -E_F
    println("Shifting DOS energies by E_F = $E_F...")
    raw_dos_up .-= E_F
    raw_dos_dn .-= E_F

    # Determine energy range and calculate broadened DOS based on *shifted* energies
    all_shifted_energies = vcat(raw_dos_up, raw_dos_dn)
    # Handle case where one spin channel might be empty (e.g., perfect FM)
    if isempty(all_shifted_energies)
        println("  Warning: No eigenvalues found for DOS calculation.")
        # Return empty results or handle as appropriate
        # Define a default grid centered around 0 (the shifted E_F)
        ω_grid = LinRange(-1.0, 1.0, dos_energy_points)
        return ω_grid, zeros(dos_energy_points), zeros(dos_energy_points)
    end
    min_E, max_E = minimum(all_shifted_energies), maximum(all_shifted_energies)
    # Define omega grid around the shifted energies, centered near 0
    ω_grid = LinRange(min_E - 5*dos_smearing_sigma, max_E + 5*dos_smearing_sigma, dos_energy_points)

    dos_up = gaussian_dos(raw_dos_up, ω_grid, dos_smearing_sigma)
    dos_dn = gaussian_dos(raw_dos_dn, ω_grid, dos_smearing_sigma)
    println("DOS calculation complete.")

    # Return the shifted energy grid and corresponding DOS
    return ω_grid, dos_up, dos_dn
end

# --------------------------------------------------------
# --- Plotting Functions ---------------------------------
# --------------------------------------------------------
function plot_bands(kdist, em_up, em_dn, title_str::String; 
    tick_positions = Float64[0.0], 
    tick_labels = String["Γ","X","M","Γ"], 
    download = true, 
    E_F = 0.0)
    """
    plot_bands(kdist, em_up, em_dn, title_str; tick_positions, tick_labels, download, E_F)

    Plots the calculated band structure.
    
    Args:
        kdist: Distance along k-path
        em_up: Spin-up band energies
        em_dn: Spin-down band energies  
        title_str: Plot title
        tick_positions: X-axis tick positions (default: [0.0])
        tick_labels: X-axis tick labels (default: ["Γ","X","M","Γ"])
        download: Whether to save plot (default: true)
        E_F: Fermi energy (default: 0.0)
    """
    p = plot(title=title_str, xlabel="k-path", ylabel="Energy (t)")
    # Check if band data is empty before trying to access size
    if isempty(em_up) && isempty(em_dn)
        println("Warning: No band data to plot for $title_str")
        return # Or plot an empty graph
    end
    # Determine num_bands safely
    num_bands = 0
    if !isempty(em_up)
        num_bands = size(em_up[1], 1)
    elseif !isempty(em_dn)
        num_bands = size(em_dn[1], 1)
    end
    
    if num_bands == 0
         println("Warning: Could not determine number of bands for $title_str")
         return
    end

    # Plot spin up bands if available
    if !isempty(em_up)
        #for band_idx in 1:num_bands
        # plot!(p, kdist, [e[band_idx] for e in em_up], label="band $band_idx ↑", color=:blue, linestyle=:solid, legend=:outertopright)
        plot!(p, kdist, [e[1] for e in em_up], label="band #1 ↑", color=:blue, linestyle=:solid, legend=:outertopright)
        plot!(p, kdist, [e[2] for e in em_up], label="band #2 ↑", color=:green, linestyle=:solid, legend=:outertopright)
        #end
    end
    # Plot spin down bands if available
    if !isempty(em_dn)
        #for band_idx in 1:num_bands
        #plot!(p, kdist, [e[band_idx] for e in em_dn], label="band $band_idx ↓", color=:red, linestyle=:dash)
        #end
        plot!(p, kdist, [e[1] for e in em_dn], label="band #1 ↓", color=:red, linestyle=:dash, legend=:outertopright)
        plot!(p, kdist, [e[2] for e in em_dn], label="band #2 ↓", color=:purple, linestyle=:dash, legend=:outertopright)

    end

    # Use provided tick positions and labels
   xticks!(p, tick_positions, tick_labels)
    
    # Set y-limits based on available data
    all_em = vcat(filter(!isempty, [em_up, em_dn])...)
    if !isempty(all_em)
        min_y = floor(minimum(minimum.(all_em))) - 1
        max_y = ceil(maximum(maximum.(all_em))) + 1
        ylims!(p, min_y, max_y)
    end
    hline!([E_F], label="E_F = $E_F", color=:black, linestyle=:dash, linewidth=2)
    display(p)
    # save plot as PNG if download is true
    download && savefig(p, "FM_band.png")
end

function plot_dos(ω_grid, dos1; dos2=[], title_str="Density of States", download = false)
    """
    plot_dos(ω_grid, dos1, dos2, title_str)

    Plots the calculated Density of States for both spins.
    The first dos argument is labeled as spin up on the plot.
    """
    # Plot spin up
    p = plot(ω_grid, dos1, label="DOS ↑", color=:blue, linewidth=2)
    xlabel!(p, "Energy (t units)")
    ylabel!(p, "ρ(ε)")
    title!(p, "$title_str (Spin Up and Spin Down)")

    # Plot spin down if given
    !isempty(dos2) && plot!(ω_grid, dos2, label="DOS ↓", color=:red, linewidth=2)
    display(p)

    download && savefig(p, "$(title_str).png")
end
