module MeanField

using LinearAlgebra
using Statistics
using Parameters
# Use '..' to access modules in the parent directory/scope
include("hubbardinit.jl") # Include the HubbardInit module
using ..HInit             

export MFParams, self_consistent_mf, compute_phase_diagram, mean_field_hamiltonian

@with_kw struct MFParams
    U::Float64            # on‑site repulsion
    t::Float64            # hopping
    ne::Float64           # electrons per site
    Nk::Int               # number of k‑points per direction
    β::Float64            # inverse temperature (for smearing)
    tol::Float64          # convergence tolerance on densities
    maxiter::Int          # max self‑consistency iterations
    maxμexpansions::Int = 10 # Max attempts to expand μ bracket
    mixing_alpha::Float64 = 0.5 # Add mixing parameter (0 < alpha <= 1)
    density_threshold::Float64 = 1e-10 # Threshold to zero out small densities
end

"""
    calculate_total_electrons(μ, p::MFParams, nup, ndown)

Calculates the total number of electrons per unit cell for a given chemical potential μ,
using the current mean-field densities nup and ndown.
"""
function calculate_total_electrons(μ, p::MFParams, nup, ndown)
    total_ne = 0.0
    # Use LinRange for periodic grid, excluding endpoint 2π
    ks = LinRange(0, 2π * (1 - 1/p.Nk), p.Nk)
    nk_tot = 0

    for kx in ks, ky in ks
        # Spin up
        Hup = mean_field_hamiltonian(kx, ky, ndown, p) # Opposite spin density
        # Ensure eigenvalues are real by treating H as Hermitian
        eps_up, _ = eigen(Hermitian(Hup))
        f_up = 1.0 ./ (exp.(p.β * (eps_up .- μ)) .+ 1.0)
        total_ne += sum(f_up) # Sum over the two bands for spin up

        # Spin down
        Hdown = mean_field_hamiltonian(kx, ky, nup, p) # Opposite spin density
        # Ensure eigenvalues are real by treating H as Hermitian
        eps_down, _ = eigen(Hermitian(Hdown))
        f_down = 1.0 ./ (exp.(p.β * (eps_down .- μ)) .+ 1.0)
        total_ne += sum(f_down) # Sum over the two bands for spin down

        nk_tot += 1
    end

    # Average over k-points (total number is Nk*Nk)
    return total_ne / (p.Nk * p.Nk) # More direct calculation
end

"""
    find_mu(p::MFParams, nup, ndown; μ_tol=1e-8, max_μ_iter=100)

Finds the chemical potential μ such that the calculated total electron density
matches the target density p.ne * 2 (for the two-site unit cell), using bisection.
"""
# Find chemical potential μ that yields target density = 2⋅ne via robust bisection with dynamic bracket expansion
function find_mu(p::MFParams, nup, ndown; μ_tol=1e-8, max_μ_iter=100)
    target_ne = p.ne * 2.0

    # initial bracket based on band edges
    μ_min = -abs(p.U) - 5*abs(p.t)
    μ_max =  abs(p.U) + 5*abs(p.t)
    ne_min = calculate_total_electrons(μ_min, p, nup, ndown)
    ne_max = calculate_total_electrons(μ_max, p, nup, ndown)

    # --- High Temperature Check ---
    # If the density at the initial low mu_min is already above the target,
    # it implies we are in a high-T regime where the target density is unreachable.
    # Return mu_min, which corresponds to the lowest achievable density.
    if ne_min > target_ne
        println("  WARNING: find_mu - Target density $target_ne is below minimum achievable density $ne_min at μ=$μ_min (β=$(p.β)). Returning μ_min.")
        return μ_min
    end
    # --- End High Temperature Check ---

    # expand bracket upwards if needed (downward expansion is now handled by the check above)
    expand = max(abs(p.t), abs(p.U)) + 1.0
    attempts = 0
    while ne_max < target_ne && attempts < p.maxμexpansions
        μ_max += expand
        ne_max = calculate_total_electrons(μ_max, p, nup, ndown)
        attempts += 1
    end

    # Check if bracketing succeeded after potential upward expansion
    if !(ne_min <= target_ne <= ne_max)
        # This error should now only trigger if upward expansion fails
        error("Density solver: failed to bracket target density $target_ne within μ range [$μ_min, $μ_max]. Results: [$ne_min, $ne_max]. Upward expansion failed?")
    end

    # now bisection
    μ_rel_tol = μ_tol * abs(μ_max - μ_min)
    final_mu = μ_mid = (μ_min + μ_max) / 2.0 # Initialize final_mu
    # Use 1:max_μ_iter directly, it's idiomatic for a fixed number of iterations
    for iter in 1:max_μ_iter
        μ_mid = (μ_min + μ_max) / 2.0
        # Check interval convergence first
        if abs(μ_max - μ_min) < μ_rel_tol
             final_mu = μ_mid
             break
        end
        ne_mid = calculate_total_electrons(μ_mid, p, nup, ndown)

        # Check density convergence
        if abs(ne_mid - target_ne) < μ_tol * target_ne
            final_mu = μ_mid
            break
        end

        if ne_mid < target_ne
            μ_min = μ_mid
        else
            μ_max = μ_mid
        end
        final_mu = (μ_min + μ_max) / 2.0 # Update final_mu for last iteration case
    end

    return final_mu
end

"""
  mean_field_hamiltonian(kx, ky, nbar, p::MFParams)

Builds the 2×2 mean‑field Hamiltonian H_kσ for spin σ at momentum k,
using ⟨n̄ασ⟩ for the opposite‑spin densities on sublattices α=1,2.
Uses standard square lattice hopping between sublattices.
"""
function mean_field_hamiltonian(kx, ky, nbar, p::MFParams)
    # Standard square lattice hopping term between sublattices (assuming a=1)
    # Note the conventional minus sign for the hopping term.
    hopping_term = -p.t * (cos(kx) + cos(ky))

    # diagonal entries: U * ⟨n̄_A⟩ and U * ⟨n̄_B⟩ (nbar[1] and nbar[2])
    H = [ p.U*nbar[1]     hopping_term
          conj(hopping_term)  p.U*nbar[2] ] # Hopping term is real here
    return H
end

"""
  self_consistent_mf(p::MFParams; init = nothing)

Runs the SCF loop to find ⟨nασ⟩ and the total energy per cell E = 
∑_{k,j,σ} f(ε_{jσ}(k)) ε_{jσ}(k)  – U ∑_α ⟨nα↑⟩⟨nα↓⟩.
Returns (n1↑,n2↑,n1↓,n2↓, E).
"""
function self_consistent_mf(p::MFParams; init=nothing)
    # initialize densities: (n1↑, n2↑, n1↓, n2↓)
    if init === nothing
        nup = fill(p.ne/2, 2)
        ndown = fill(p.ne/2, 2)
    else
        nup, ndown = init
    end

    # prepare k‑grid
    # Use LinRange for periodic grid, excluding endpoint 2π
    ks = LinRange(0, 2π * (1 - 1/p.Nk), p.Nk)
    µ = 0.0  # Initialize mu

    # Use 1:p.maxiter directly
    for iter in 1:p.maxiter
        nup_old, ndown_old = copy(nup), copy(ndown) # Use different variable names
        nk = 0

        # Determine chemical potential µ using the solver
        µ = find_mu(p, nup, ndown) # Pass current densities

        # reset accumulators
        nup_new = zeros(Float64, 2)
        ndown_new = zeros(Float64, 2)
        Eband = 0.0

        for kx in ks, ky in ks
            # spin ↑: opposite densities = ndown
            Hup = mean_field_hamiltonian(kx,ky, ndown, p)
            # Ensure eigenvalues are real by treating H as Hermitian
            eps_up, Vup = eigen(Hermitian(Hup))
            # spin ↓: opposite densities = nup
            Hdown = mean_field_hamiltonian(kx,ky, nup, p)
            # Ensure eigenvalues are real by treating H as Hermitian
            eps_down, Vdown = eigen(Hermitian(Hdown))

            for (εs, Vs, nacc) in ((eps_up,Vup,nup_new),(eps_down,Vdown,ndown_new))
                # Apply chemical potential shift here for Fermi-Dirac
                f = 1.0 ./ (exp.(p.β*(εs .- µ)) .+ 1.0)
                # accumulate band energy
                Eband += sum(εs .* f) # Sum energies weighted by occupation
                #  |V[α,j]|^2 gives contribution of band j to sublattice α
                # Use axes for robustness, although size is fixed at 2x2 here
                for α in axes(Vs, 1), j in axes(Vs, 2)
                    nacc[α] += abs2(Vs[α,j]) * f[j]
                end
            end
            nk += 1
        end

        # normalize by number of k‑points
        nup_new ./= nk; ndown_new ./= nk
        Eband /= nk

        # --- Zero out densities below threshold ---
        threshold = p.density_threshold
        # Use eachindex for iterating over density arrays
        for i in eachindex(nup_new)
            if abs(nup_new[i]) < threshold
                nup_new[i] = 0.0
            end
        end
        for i in eachindex(ndown_new)
             if abs(ndown_new[i]) < threshold
                ndown_new[i] = 0.0
            end
        end
        # --- End zeroing ---

        # --- Apply Linear Mixing ---
        alpha = p.mixing_alpha
        nup .= alpha .* nup_new .+ (1.0 - alpha) .* nup_old
        ndown .= alpha .* ndown_new .+ (1.0 - alpha) .* ndown_old
        # --- End Mixing ---

        # total energy adds the mean‑field “double‑counting” term –U ∑α⟨nα↑⟩⟨nα↓⟩
        Eint = -p.U * sum(nup .* ndown)
        Etot = Eband + Eint

        # check convergence
        if maximum(abs.([nup .- nup_old; ndown .- ndown_old])) < p.tol # Compare with nup_old, ndown_old
            println("SCF converged in $iter iterations. μ = $µ") # Added convergence message with μ
            return nup, ndown, Etot
        end
    end
    error("SCF did not converge in $(p.maxiter) iterations")
end

"""
  compute_phase_diagram(ps::Vector{MFParams})

Loops over a grid of (ne, U/t) to classify each point as PM, FM or AFM by comparing
ground‑state energies under the three trial configurations:
  • Paramagnetic:  n1↑=n2↑=n1↓=n2↓=ne/2
  • Ferromagnetic: e.g. n1↑=n2↑=ne,  n1↓=n2↓=0
  • Antiferromagnetic: n1↑=n2↓=nup, n1↓=n2↑=ndown
Returns a matrix of phase labels.
"""
function compute_phase_diagram(ps::Vector{MFParams})
    # Define phase mapping using integers directly if preferred
    phases = Dict(:PM=>1, :FM=>2, :AFM=>3) 
    phase_indices = [:PM, :FM, :AFM] # Map index back to symbol if needed

    grid = fill(0, length(ps))
    # Use eachindex for iterating over the parameter vector
    for (i,p) in pairs(ps) # Use pairs for index and value
        println("Calculating point $(i)/$(length(ps)): ne=$(p.ne), U=$(p.U)") # Progress
        # PM
        p_PM = p # No need to copy MFParams if it's immutable or not modified
        nup_PM = fill(p.ne/2, 2); ndown_PM = copy(nup_PM) # Use copy if needed later
        _,_,E_PM = self_consistent_mf(p_PM; init=(nup_PM, ndown_PM))

        # FM
        nup_FM = fill( p.ne, 2); ndown_FM = fill(0.0, 2) # Use 0.0
        _,_,E_FM = self_consistent_mf(p; init=(nup_FM, ndown_FM))

        # AFM
        δ = 0.1 # Define delta for AFM seed
        nup_AFM = [p.ne/2+δ, p.ne/2-δ]   
        ndown_AFM = [p.ne/2-δ, p.ne/2+δ]
        _,_,E_AFM = self_consistent_mf(p; init=(nup_AFM, ndown_AFM))

        # pick lowest energy and assign phase index
        Emin, phase_index = findmin([E_PM, E_FM, E_AFM])
        grid[i] = phase_index # Assign the index (1, 2, or 3) directly
        println("  -> Phase: $(phase_indices[phase_index]), E_PM=$E_PM, E_FM=$E_FM, E_AFM=$E_AFM")
    end
    return grid
end

end # module

# ----------------------------------------------------------------------------
# Test the module

using Test

