# fig53.jl
# Refactored script to reproduce Figure 5 with decompartmentalized functions
# Allows separate calculation of bands and DOS.

using LinearAlgebra      # norm, Hermitian
using Plots              # plotting backend
using Statistics         # basic stats
using Printf             # For formatting output

# Load the original modules
include("src/hubbardinit.jl")
include("src/reciprocal.jl")
include("src/meanfield.jl")

using .MeanField          # bring mean-field functions into scope
using .ReciprocalSpace    # bring reciprocal space constants/functions into scope

# --- Helper Functions --- 

"""
    define_k_path(Nk)

Defines the k-path Γ→X→M→Γ for band structure plots.
Returns the path coordinates and the cumulative distance along the path.
"""
function define_k_path(Nk)
    ks1 = [(kx, 0.0) for kx in LinRange(0, π, Nk)]
    ks2 = [(π, ky)  for ky in LinRange(0, π, Nk)]
    ks3 = [(kx, kx) for kx in LinRange(π, 0, Nk)]
    path = vcat(ks1[1:end-1], ks2[1:end-1], ks3)
    d = [0.0; [norm(path[i] .- path[i-1]) for i in eachindex(path)[2:end]]]
    kdist = cumsum(d)
    return path, kdist
end

"""
    gaussian_dos(energies, ω_grid, σ)

Calculates the Density of States (DOS) using Gaussian broadening.
(Same as in fig52.jl)
"""
function gaussian_dos(energies, ω_grid, σ)
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

# --- Core Calculation Functions --- 

"""
    run_scf(params::MFParams; initial_guess=nothing)

Runs the self-consistent field calculation.
Returns converged densities (nup, ndown) and total energy.
"""
function run_scf(params::MFParams; initial_guess=nothing)
    println("Starting SCF for U=$(params.U), ne=$(params.ne)...")
    nup, ndown, Etot = self_consistent_mf(params; init=initial_guess)
    println("Converged densities: n↑=", nup, ", n↓=", ndown, ", E=$Etot")
    return nup, ndown, Etot
end

"""
    calculate_bands(params::MFParams, nup, ndown, k_path)

Calculates band structure eigenvalues along a given k-path.
Returns eigenvalues for spin up (em_up) and spin down (em_dn).
"""
function calculate_bands(params::MFParams, nup, ndown, k_path)
    println("Calculating band structure...")
    evals_up(p, ndn) = [eigvals(Hermitian(mean_field_hamiltonian(kx, ky, ndn, p))) for (kx, ky) in k_path]
    evals_dn(p, nup) = [eigvals(Hermitian(mean_field_hamiltonian(kx, ky, nup, p))) for (kx, ky) in k_path]
    
    em_up = evals_up(params, ndown)
    em_dn = evals_dn(params, nup)
    println("Band structure calculation complete.")
    return em_up, em_dn
end

"""
    calculate_dos(params::MFParams, nup, ndown; 
                  Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)

Calculates the Density of States (DOS).
Returns the energy grid (ω_grid) and DOS for spin up/down (dos_up, dos_dn).
"""
function calculate_dos(params::MFParams, nup, ndown; 
                       Nk_dos=500, dos_smearing_sigma=0.05, dos_energy_points=400)
    
    println("\nCalculating DOS on $Nk_dos x $Nk_dos grid...")
    kx_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    ky_grid = LinRange(0, 2π * (1 - 1/Nk_dos), Nk_dos)
    grid = [(kx, ky) for kx in kx_grid, ky in ky_grid]

    # Calculate raw eigenvalues for DOS
    println("  Calculating raw eigenvalues...")
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

    println("  Calculating broadened DOS (σ = $dos_smearing_sigma)...")
    dos_up = gaussian_dos(raw_dos_up, ω_grid, dos_smearing_sigma)
    dos_dn = gaussian_dos(raw_dos_dn, ω_grid, dos_smearing_sigma)
    println("DOS calculation complete.")

    return ω_grid, dos_up, dos_dn
end

# --- Plotting Functions --- (Identical to fig52.jl)

"""
    plot_bands(kdist, em_up, em_dn, title_str, filename)

Plots the calculated band structure.
"""
function plot_bands(kdist, em_up, em_dn, title_str, filename)
    p = plot(title=title_str, xlabel="k-path", ylabel="Energy (t units)")
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

    # Add k-path labels
    Nk = length(kdist) ÷ 3 + 1 # Infer Nk from kdist length
    tick_indices = [1, Nk, 2*(Nk-1)+1, length(kdist)]
    tick_positions = kdist[tick_indices]
    xticks!(p, tick_positions, ["Γ","X","M","Γ"])
    
    # Set y-limits based on available data
    all_em = vcat(filter(!isempty, [em_up, em_dn])...)
    if !isempty(all_em)
        min_y = floor(minimum(minimum.(all_em))) - 1
        max_y = ceil(maximum(maximum.(all_em))) + 1
        ylims!(p, min_y, max_y)
    end
    
    display(p)
    savefig(p, filename)
end

"""
    plot_dos(ω_grid, dos_up, dos_dn, title_str, filename_prefix)

Plots the calculated Density of States for both spins.
"""
function plot_dos(ω_grid, dos_up, dos_dn, title_str, filename_prefix)
    # Plot spin up
    p_up = plot(ω_grid, dos_up, label="DOS ↑", color=:red, linewidth=2)
    xlabel!(p_up, "Energy (t units)")
    ylabel!(p_up, "ρ(ε)")
    title!(p_up, "$title_str (Spin Up and Spin Down)")
    savefig(p_up, "$(filename_prefix)_up.png")

    # Plot spin down
    plot!(ω_grid, dos_dn, label="DOS ↓", color=:blue, linewidth=2)
    display(p_up)
    savefig(p_up, "$(filename_prefix)_dn.png")
end

# --- Main Execution --- 

function main(; compute_dos::Bool = true) # Add flag to control DOS calculation
    println("Starting calculations...")

    # --- Parameters --- 
    t = 1.0
    ne = 1.6
    Ufm_val = t / 0.077
    Uafm_val = t / 0.2
    Nk_scf = 50       # K-points for SCF convergence
    Nk_bands = 50     # K-points for band structure path
    Nk_dos_grid = 500 # K-points per dim for DOS grid
    beta = 1.0
    scf_tol = 1e-6
    scf_maxiter = 200
    dos_sigma = 0.05
    dos_points = 400

    # --- Setup MFParams --- 
    p_fm  = MFParams(U=Ufm_val,  t=t, ne=ne, Nk=Nk_scf, β=beta, tol=scf_tol, maxiter=scf_maxiter)
    p_afm = MFParams(U=Uafm_val, t=t, ne=ne, Nk=Nk_scf, β=beta, tol=scf_tol, maxiter=scf_maxiter)

    # --- Initial Guesses --- 
    #=
    delta = 0.1 # Small symmetry breaking 
    nup_fm_init = [p_fm.ne/4 + delta, p_fm.ne/4 - delta]
    ndown_fm_init = [p_fm.ne/4 - delta, p_fm.ne/4 + delta]
    nup_afm_init = [p_afm.ne/4 + delta, p_afm.ne/4 - delta]
    ndown_afm_init = [p_afm.ne/4 - delta, p_afm.ne/4 + delta] 

    # --- Run SCF --- 
    nup_fm, ndown_fm, E_fm = run_scf(p_fm, initial_guess=(nup_fm_init, ndown_fm_init))
    nup_afm, ndown_afm, E_afm = run_scf(p_afm, initial_guess=(nup_afm_init, ndown_afm_init))
    =#

    # --- Calculate Band Structure --- 
    # --- for ferromagnetic and antiferromagnetic states ---
    nup_fm = fill(0.8, 2)
    ndown_fm = fill(0, 2)
    nup_afm = [0.62, 0.18]
    ndown_afm = [0.18, 0.62]

    k_path, k_dist = define_k_path(Nk_bands) 
    #=
    em_fm_up, em_fm_dn = calculate_bands(p_fm, nup_fm, ndown_fm, k_path)
    em_afm_up, em_afm_dn = calculate_bands(p_afm, nup_afm, ndown_afm, k_path)

    # --- Plot Band Structure --- 
    plot_bands(k_dist, em_fm_up, em_fm_dn, "FM Mean-field Bands (fig53)", "fig54_FM_bands.png")
    plot_bands(k_dist, em_afm_up, em_afm_dn, "AFM Mean-field Bands (fig53)", "fig54_AFM_bands.png")
    =#
    # --- Optionally Calculate and Plot DOS --- 
    if compute_dos
        # FM DOS
        ω_grid_fm, dos_fm_up, dos_fm_dn = calculate_dos(p_fm, nup_fm, ndown_fm; 
                                                      Nk_dos=Nk_dos_grid, 
                                                      dos_smearing_sigma=dos_sigma, 
                                                      dos_energy_points=dos_points)
        plot_dos(ω_grid_fm, dos_fm_up, dos_fm_dn, "FM Density of States ", "FM_dos")

        # AFM DOS
        ω_grid_afm, dos_afm_up, dos_afm_dn = calculate_dos(p_afm, nup_afm, ndown_afm; 
                                                         Nk_dos=Nk_dos_grid, 
                                                         dos_smearing_sigma=dos_sigma, 
                                                         dos_energy_points=dos_points)
        plot_dos(ω_grid_afm, dos_afm_up, dos_afm_dn, "AFM Density of States ", "AFM_dos")
    else
        println("Skipping DOS calculation and plotting.")
    end

    println("Script finished.")
end

# --- Run Main Function --- 
# Call main() to run everything including DOS
# Call main(compute_dos=false) to run only SCF and Bands
main(compute_dos=true) 
