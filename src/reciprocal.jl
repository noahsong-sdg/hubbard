# functions that compute bands and dos' for various occupation levels 
module ReciprocalSpace

using LinearAlgebra      # norm, Hermitian
using Plots              
using Statistics         
using Printf             # For formatting output

using ..HInit             # Needed for HubbardParams in calculate_dispersion
using ..MeanField          # bring mean-field functions into scope

export calculate_dispersion, 
       generate_k_path, 
       plot_dispersion, 
       plot_bands,
       gamma_k,
       get_bands,
       # Export the constants
       a, b1, b2, k,
       t, Γ, X, M, KPATH,
       plot_band_structure,
       calculate_bands, calculate_dos,
       gaussian_dos, 
       plot_dos, plot_bands


# ----------------------------------------------------------------------------- #

# Constants related to reciprocal space
const a = 1.0  # Lattice constant
const b1 = [1.0, 1.0]
const b2 = [1.0, -1.0]
const k = [0, 0]  # Default k-point
const t = 1.0  # Default hopping parameter

const Γ = [0.0, 0.0]
const X = [π, 0.0]
const M = [π, π]
const KPATH = [Γ, X, M, Γ]
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
    point_labels = ["Γ", "X", "M", "Γ", "Y", "Z"] # Extend as needed
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

gamma_k(kvec, b1, b2) = -(1 + exp(-im * dot(kvec, b1)) + exp(-im * dot(kvec, b2)) + exp(-im * dot(kvec, b1 + b2)))

function gaussian_dos(energies, ω_grid, σ)
    """
    gaussian_dos(energies, ω_grid, σ)

    Calculates the Density of States (DOS) using Gaussian broadening.
    (Same as in fig52.jl)
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
    Nk_sq = round(Int, sqrt(length(energies)/2)) # Infer Nk_dos*Nk_dos
    if Nk_sq > 0
         dos ./= Nk_sq # Normalize by number of k-points
    end
    return dos
end
# ---------------------------------------------------
# --- Core Calculation Functions --- ------------------
# --------------------------------------------------
function calculate_bands(params::MFParams, nup, ndown, k_path)
    """
    calculate_bands(params::MFParams, nup, ndown, k_path)

    Calculates band structure eigenvalues along a given k-path.
    Returns eigenvalues for spin up (em_up) and spin down (em_dn).
    """
    println("Calculating band structure...")
    evals_up(p, ndn) = [eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndn, p))) for (kx, ky) in k_path]
    evals_dn(p, nup) = [eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, p))) for (kx, ky) in k_path]
    
    em_up = evals_up(params, ndown)
    em_dn = evals_dn(params, nup)
    println("Band structure calculation complete.")
    return em_up, em_dn
end

function calculate_dos(params::MFParams, nup, ndown; 
                       Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)
    """
    calculate_dos(params::MFParams, nup, ndown; 
                  Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)

    Calculates the Density of States given occupation numbers and standard parameter set (DOS).
    Returns the energy grid (ω_grid) and DOS for spin up/down (dos_up, dos_dn).
    """
    println("\nCalculating DOS on $Nk_dos x $Nk_dos grid...")
    # generate the Brillouin zone grid
    kx_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    ky_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    grid = [(kx, ky) for kx in kx_grid, ky in ky_grid]

    # Calculate raw eigenvalues for DOS
    println("Calculating eigenvalues...")
    raw_dos_up_func(p, ndn) = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndn, p))) for (kx, ky) in grid]...)
    raw_dos_dn_func(p, nup) = vcat([eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, p))) for (kx, ky) in grid]...)
    raw_dos_up = raw_dos_up_func(params, ndown)
    raw_dos_dn = raw_dos_dn_func(params, nup)

    # Determine energy range and calculate broadened DOS
    all_energies = vcat(raw_dos_up, raw_dos_dn)
    # Handle case where one spin channel might be empty (e.g., perfect FM)
    if isempty(all_energies)
        println("  Warning: No eigenvalues found for DOS calculation.")
        # Return empty results or handle as appropriate
        ω_grid = LinRange(0, 1, dos_energy_points) # Default grid
        return ω_grid, zeros(dos_energy_points), zeros(dos_energy_points)
    end
    min_E, max_E = minimum(all_energies), maximum(all_energies)
    ω_grid = LinRange(min_E - 5*dos_smearing_sigma, max_E + 5*dos_smearing_sigma, dos_energy_points)

    println("Calculating broadened DOS (σ = $dos_smearing_sigma)...")
    dos_up = gaussian_dos(raw_dos_up, ω_grid, dos_smearing_sigma)
    dos_dn = gaussian_dos(raw_dos_dn, ω_grid, dos_smearing_sigma)
    println("DOS calculation complete.")

    return ω_grid, dos_up, dos_dn
end

# --------------------------------------------------------
# --- Plotting Functions ---------------------------------
# --------------------------------------------------------
function plot_bands(kdist, em_up, em_dn, title_str,  tick_positions, tick_labels)
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
    
    display(p)
    savefig(p, title_str * ".png")
end

function plot_dos(ω_grid, dos1, dos2=[], title_str="Density of States")
    """
    plot_dos(ω_grid, dos1, dos2, title_str)

    Plots the calculated Density of States for both spins.
    The first dos argument is labeled as spin up on the plot.
    """
    # Plot spin up
    p = plot(ω_grid, dos1, label="DOS ↑", color=:red, linewidth=2)
    xlabel!(p, "Energy (t units)")
    ylabel!(p, "ρ(ε)")
    title!(p, "$title_str (Spin Up and Spin Down)")

    # Plot spin down if given
    !isempty(dos2) && plot!(ω_grid, dos2, label="DOS ↓", color=:blue, linewidth=2)
    display(p)
    savefig(p, "$(title_str).png")
end



end
