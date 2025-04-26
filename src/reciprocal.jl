# functions that compute bands and dos' for various occupation levels 
module ReciprocalSpace

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf
using Statistics

using ..HInit             # Needed for HubbardParams in calculate_dispersion
using ..MeanField          # bring mean-field functions into scope

export calculate_dispersion, 
       generate_k_path, 
       plot_dispersion, 
       gamma_k,
       calculate_bands, calculate_dos,
       calculate_fermi_level, # Add new function
       gaussian_dos, 
       plot_dos, plot_bands

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
    supercell::Bool = false
)
    """
    generate_k_path(; points, labels, num_points_per_segment=50, supercell=false)

    - If `points` is empty, picks the default path:
        • supercell=false: Γ–X–M–Γ
        • supercell=true:  Γ_b–M_b–Γ_b–X_b–Γ_b
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

#=
function generate_k_path(points, num_points_per_segment=50)
    """
    generate_k_path(points, num_points_per_segment=50)

    Generates a list of k-points interpolating linearly between the given `points`.
    Also calculates the cumulative distance along the path.

    Args:
        points: A list of k-space coordinates (e.g., [[0,0], [π,0], [π,π], [0,0]]).
        num_points_per_segment: Number of points to generate for each segment between consecutive points in `points`.

    Returns:
        k_path: A vector of k-point coordinates (tuples).
        k_dist: A vector of cumulative distances corresponding to each point in `k_path`.
        tick_positions: A vector of distances corresponding to the input high-symmetry `points`.
        tick_labels: A vector of labels for the high-symmetry points (currently hardcoded for Γ, X, M).
    """ 
    k_path = []
    k_dist = [0.0] # Start distance at 0
    tick_positions = [0.0]
    # Simple labeling for now, could be made more general
    # Adjust labels based on the input path points
    if points == KPATH
        point_labels = ["Γ", "X", "M", "Γ"]
    elseif points == KPATH_B
        point_labels = ["Γ_b", "X_b", "Γ_b", "M_b", "Γ_"] # Use primes for AFM path points
    else
        point_labels = ["P$i" for i in 1:length(points)] # Generic labels
    end
    tick_labels = [point_labels[i] for i in 1:length(points)]

    current_dist = 0.0
    # Iterate through segments defined by consecutive points
    for i in 1:(length(points)-1)
        p1 = points[i]
        p2 = points[i+1]
        segment_vec = p2 .- p1
        segment_len = norm(segment_vec)

        # Generate points along the segment using LinRange
        # Exclude the last point (t=1.0) to avoid duplication with the start of the next segment
        num_steps = (i == length(points) - 1) ? num_points_per_segment : num_points_per_segment -1
        for step in 0:num_steps
            t = step / num_points_per_segment
            k = p1 .+ t .* segment_vec
            # Add the first point of the first segment, or points from subsequent steps
            if i == 1 && step == 0
                push!(k_path, tuple(k...))
            elseif step > 0
                 push!(k_path, tuple(k...))
                 # Calculate distance from the previous point in the path
                 dist_step = norm(k .- k_path[end-1])
                 current_dist += dist_step
                 push!(k_dist, current_dist)
            end
        end
        # Store the position of the high-symmetry point at the end of the segment
        push!(tick_positions, current_dist)
    end

    # Ensure k_dist has the same length as k_path
    if length(k_dist) != length(k_path)
       println("Warning: k_dist and k_path length mismatch. Recalculating k_dist.")
       k_dist = [0.0]
       for idx = 2:length(k_path)
           push!(k_dist, k_dist[end] + norm(k_path[idx] .- k_path[idx-1]))
       end
    end

    return k_path, k_dist, tick_positions, tick_labels
end
=#
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

    Returns eigenvalues for spin up (em_up) and spin down (em_dn), shifted if E_F is provided.
    """
    println("Calculating band structure...")
    # Note: Pass n_opposite_spin to mean_field_hamiltonian
    evals_up(p, ndn) = [sort(eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndn, p)))) for (kx, ky) in k_path]
    evals_dn(p, nup) = [sort(eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, p)))) for (kx, ky) in k_path]

    em_up = evals_up(params, ndown)
    em_dn = evals_dn(params, nup)

    # Shift energies by -E_F
    shifted_em_up = [[e - E_F for e in evals] for evals in em_up] # Equivalent list comprehension
    shifted_em_dn = [[e - E_F for e in evals] for evals in em_dn] # Equivalent list comprehension

    println("Band structure calculation complete. Energies shifted by E_F = $E_F.")
    return shifted_em_up, shifted_em_dn
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

    println("Calculating broadened DOS (σ = $dos_smearing_sigma) with shifted energies...")
    # Use shifted energies to calculate DOS on the new grid
    dos_up = gaussian_dos(raw_dos_up, ω_grid, dos_smearing_sigma)
    dos_dn = gaussian_dos(raw_dos_dn, ω_grid, dos_smearing_sigma)
    println("DOS calculation complete.")

    # Return the shifted energy grid and corresponding DOS
    return ω_grid, dos_up, dos_dn
end

# --------------------------------------------------------
# --- Plotting Functions ---------------------------------
# --------------------------------------------------------
function plot_bands(kdist, em_up, em_dn, title_str::String, 
    tick_positions, tick_labels; download = true, E_F = 0.0)
    """
    plot_bands(kdist, em_up, em_dn, title_str, tick_positions, tick_labels)

    Plots the calculated band structure.
    Accepts tick positions and labels for the x-axis.
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
        for band_idx in 1:num_bands
            plot!(p, kdist, [e[band_idx] for e in em_up], label="band $band_idx ↑", color=:blue, linestyle=:solid, legend=:outertopright)
        end
    end
    # Plot spin down bands if available
    if !isempty(em_dn)
        for band_idx in 1:num_bands
            plot!(p, kdist, [e[band_idx] for e in em_dn], label="band $band_idx ↓", color=:red, linestyle=:dash)
        end
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



end
