module MeanField

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf
using Statistics
using Parameters

# Use '..' to access modules in the parent directory/scope
using ..HInit             

export MFParams, self_consistent_mf, compute_phase_diagram, mean_field_hamiltonian

@with_kw struct MFParams
    U::Float64            # on‑site repulsion
    t::Float64            # hopping
    ne::Float64           # electrons per site (target filling for the 2-site unit cell is 2*ne)
    Nk::Int               # number of k‑points per direction for BZ integration
    β::Float64            # inverse temperature (for Fermi-Dirac smearing)
    tol::Float64          # convergence tolerance on sublattice densities
    maxiter::Int          # max self‑consistency iterations
    maxμexpansions::Int = 10 # Max attempts to expand μ bracket in find_mu
    mixing_alpha::Float64 = 0.5 # Linear mixing parameter for density updates (0 < alpha <= 1)
    density_threshold::Float64 = 1e-10 # Threshold below which densities are zeroed out
end

function calculate_total_electrons(μ, p::MFParams, nup, ndown)
    """
    calculate_total_electrons(μ, p::MFParams, nup, ndown)

    Calculates the total number of electrons per unit cell (2 sites) for a given chemical potential `μ`,
    using the current mean-field densities `nup` (⟨n₁↑⟩, ⟨n₂↑⟩) and `ndown` (⟨n₁↓⟩, ⟨n₂↓⟩).
    Integrates the Fermi-Dirac distribution over the Brillouin zone using `p.Nk` x `p.Nk` k-points.

    Args:
        μ (Float64): The chemical potential.
        p (MFParams): Mean-field parameters.
        nup (Vector{Float64}): Current spin-up densities on sublattices [A, B].
        ndown (Vector{Float64}): Current spin-down densities on sublattices [A, B].

    Returns:
        Float64: The calculated total electron density per unit cell (ranging from 0 to 4).
    """
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


function find_mu(p::MFParams, nup, ndown; μ_tol=1e-8, max_μ_iter=100)
    """
    find_mu(p::MFParams, nup, ndown; μ_tol=1e-8, max_μ_iter=100)

    Finds the chemical potential `μ` such that the calculated total electron density per unit cell
    matches the target density `p.ne * 2`, using the current mean-field densities `nup` and `ndown`.
    Employs a robust bisection method with dynamic bracket expansion.

    Args:
        p (MFParams): Mean-field parameters.
        nup (Vector{Float64}): Current spin-up densities on sublattices [A, B].
        ndown (Vector{Float64}): Current spin-down densities on sublattices [A, B].
        μ_tol (Float64, optional): Tolerance for convergence on `μ` or density. Defaults to 1e-8.
        max_μ_iter (Int, optional): Maximum iterations for the bisection search. Defaults to 100.

    Returns:
        Float64: The converged chemical potential `μ`.

    Raises:
        Error: If the target density cannot be bracketed within the expanded `μ` range.
    """
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

gamma_k(kvec) = -(1 + exp(-im * dot(kvec, b1)) + exp(-im * dot(kvec, b2)) + exp(-im * dot(kvec, b1 + b2)))

function mean_field_hamiltonian(kx, ky, nbar, p::MFParams)
    """
  mean_field_hamiltonian(kx, ky, nbar, p::MFParams)

    Builds the 2×2 mean‑field Hamiltonian `H_kσ` for a given spin `σ` at momentum `k = (kx, ky)`,
    using the opposite-spin densities `nbar = ⟨n̄₁, n̄₂⟩` on the two sublattices (A=1, B=2).
    Assumes a square lattice with nearest-neighbor hopping `t` and on-site interaction `U`.

    Args:
        kx (Float64): k-vector x-component.
        ky (Float64): k-vector y-component.
        nbar (Vector{Float64}): Opposite-spin densities on sublattices [A, B].
        p (MFParams): Mean-field parameters (uses `t` and `U`).

    Returns:
        Matrix{Float64}: The 2x2 mean-field Hamiltonian for the specified spin and k-point.
    """
    # Standard square lattice hopping term between sublattices (assuming a=1)
    # Note the conventional minus sign for the hopping term.
    offdiag = p.t * gamma_k([kx, ky]) # Divide by 4 for two sites

    # diagonal entries: U * ⟨n̄_A⟩ and U * ⟨n̄_B⟩ (nbar[1] and nbar[2])
    H = [ p.U*nbar[1]     offdiag
          conj(offdiag)  p.U*nbar[2] ] # Hopping term is real here
    return H
end


function self_consistent_mf(p::MFParams; init=nothing)
    """
  self_consistent_mf(p::MFParams; init = nothing)

    Performs the self-consistent field (SCF) calculation for the two-site Hubbard model
    within the mean-field approximation.
    Iteratively updates the sublattice densities (n↑₁, n↑₂, n↓₁, n↓₂) until convergence or max iterations.
    Calculates the chemical potential `μ` at each step to enforce the target filling `p.ne`.

    Args:
        p (MFParams): Structure containing all parameters for the calculation.
        init (Tuple{Vector{Float64}, Vector{Float64}}, optional): Initial guess for (nup, ndown).
            If `nothing`, starts with a paramagnetic guess based on `p.ne`. Defaults to `nothing`.

    Returns:
        Tuple{Vector{Float64}, Vector{Float64}, Float64}: A tuple containing:
            - `nup`: Converged spin-up densities [⟨n₁↑⟩, ⟨n₂↑⟩].
            - `ndown`: Converged spin-down densities [⟨n₁↓⟩, ⟨n₂↓⟩].
            - `Etot`: Converged total energy per unit cell, including the double-counting correction.

    Raises:
        Error: If the SCF loop does not converge within `p.maxiter` iterations.
    """
    println("Starting SCF for U=$(params.U), ne=$(params.ne)...")
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
            println("Converged densities: n↑=", nup, ", n↓=", ndown, ", E=$Etot")
            return nup, ndown, Etot
        end
    end
    error("SCF did not converge in $(p.maxiter) iterations")
end


function compute_phase_diagram(ps::Vector{MFParams})
    """
  compute_phase_diagram(ps::Vector{MFParams})

    Computes the ground state phase (Paramagnetic, Ferromagnetic, or Antiferromagnetic)
    for each set of parameters in the input vector `ps`.
    For each parameter set, it runs SCF calculations starting from PM, FM, and AFM initial guesses
    and determines the phase with the lowest converged total energy.

    Args:
        ps (Vector{MFParams}): A vector where each element is an `MFParams` struct defining
            a point in the parameter space (e.g., varying U and ne).

    Returns:
        Vector{Int}: A vector of the same length as `ps`, where each element is an integer
                    representing the ground state phase:
                    - 1: Paramagnetic (PM)
                    - 2: Ferromagnetic (FM)
                    - 3: Antiferromagnetic (AFM)
    """
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

