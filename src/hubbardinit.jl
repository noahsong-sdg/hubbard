module HInit

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf
using Statistics

export HubbardParams, 
       create_basis, 
       build_hamiltonian, 
       calculate_site_occupations, 
       calculate_spin_correlation,
       plot_occupations, 
       plot_spin_correlation,
       calculate_double_occupancy,
       calculate_charge_gap,
       validate_mott_transition,
       plot_mott_transition,
       analytical_charge_gap,
       analytical_double_occupancy,
       calculate_critical_point,
       calculate_dispersion,
       a, b1, b2, k,
       t, Γ, X, M, KPATH, KPATH_B


# Constants related to reciprocal space
const a = 1.0  # Lattice constant
const b1 = [1.0, 1.0]
const b2 = [1.0, -1.0]
const k = [0, 0]  # Default k-point
const t = 1.0  # Default hopping parameter

#edge = pi / sqrt(2)
const Γ = [0.0, 0.0]
const X = [pi, 0.0]
const M = [pi, pi] 
const KPATH = [Γ, X, M, Γ]

# High-symmetry points for the original Brillouin Zone

# High-symmetry points for the magnetic Brillouin Zone (AFM state)
# These correspond to folding the original BZ.
const Γ_b = [0.0, 0.0]   # Same as Γ
const X_b = [π, π] # Center of the edge of the original BZ face
const M_b = [π, 0.0]   # Same as X in the original BZ
const KPATH_B = [Γ_b, X_b, Γ_b, M_b, Γ_b] # Path for AFM state

# System parameters
struct HubbardParams
    L::Int      # Number of sites
    N_up::Int   # Number of up-spin electrons
    N_dn::Int   # Number of down-spin electrons
    t::Float64  # Hopping strength
    U::Float64  # On-site interaction
end

"""
    create_basis(L::Int, N::Int)

Generates the basis states for `N` electrons on `L` sites using integer representation.
Each integer represents a configuration where the k-th bit being set means site k is occupied.
Uses Gosper's hack to efficiently generate combinations.

Args:
    L (Int): Total number of sites.
    N (Int): Number of electrons.

Returns:
    Vector{Int}: A vector of integers, each representing a basis state.
                 Returns an empty vector if N > L or N < 0.
"""
function create_basis(L, N)
    if N > L || N < 0
        return Int[]  # Invalid parameters
    end
    
    num_states = binomial(L, N)
    basis = Vector{Int}(undef, num_states)
    
    if N == 0
        basis[1] = 0  # No electrons means all zeros
        return basis
    end
    
    state = (1 << N) - 1
    idx = 1
    basis[idx] = state
    
    while idx < num_states
        x = state & -state
        y = state + x
        state = (((state & ~y) ÷ x) >> 1) | y
        
        idx += 1
        basis[idx] = state
    end
    
    return basis
end

"""
    build_hamiltonian(params::HubbardParams)

Constructs the Hubbard Hamiltonian matrix in the occupation basis for the given parameters.
Includes kinetic hopping terms (with periodic boundary conditions) and the on-site interaction term.

Args:
    params (HubbardParams): Structure containing system parameters (L, N_up, N_dn, t, U).

Returns:
    Matrix{Float64}: The dense Hubbard Hamiltonian matrix.
                     Note: For larger systems, a sparse representation is recommended.
"""
function build_hamiltonian(params::HubbardParams)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    dim = length(up_basis) * length(dn_basis)
    H = zeros(Float64, dim, dim)
    
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            for site in 1:params.L
                next_site = site % params.L + 1
                
                if (up & (1 << (site-1))) != 0 && (up & (1 << (next_site-1))) == 0
                    new_up = up ⊻ (1 << (site-1)) ⊻ (1 << (next_site-1))
                    j_up = findfirst(==(new_up), up_basis)
                    if j_up !== nothing
                        idx1 = (i_up-1)*length(dn_basis) + i_dn
                        idx2 = (j_up-1)*length(dn_basis) + i_dn
                        H[idx1, idx2] -= params.t
                        H[idx2, idx1] -= params.t
                    end
                end
                
                if (dn & (1 << (site-1))) != 0 && (dn & (1 << (next_site-1))) == 0
                    new_dn = dn ⊻ (1 << (site-1)) ⊻ (1 << (next_site-1))
                    j_dn = findfirst(==(new_dn), dn_basis)
                    if j_dn !== nothing
                        idx1 = (i_up-1)*length(dn_basis) + i_dn
                        idx2 = (i_up-1)*length(dn_basis) + j_dn
                        H[idx1, idx2] -= params.t
                        H[idx2, idx1] -= params.t
                    end
                end
            end
            
            for site in 1:params.L
                if (up & (1 << (site-1))) != 0 && (dn & (1 << (site-1))) != 0
                    idx = (i_up-1)*length(dn_basis) + i_dn
                    H[idx, idx] += params.U
                end
            end
        end
    end
    return H
end

"""
    calculate_site_occupations(ψ::Vector{ComplexF64}, params::HubbardParams)

Calculates the expected site occupations for spin-up, spin-down, and total density
from a given wavefunction `ψ`.

Args:
    ψ (Vector{ComplexF64}): The wavefunction vector in the combined basis.
    params (HubbardParams): Structure containing system parameters.

Returns:
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}: A tuple containing:
        - `up_occupation`: Vector of expected occupations for spin-up electrons at each site.
        - `dn_occupation`: Vector of expected occupations for spin-down electrons at each site.
        - `total_occupation`: Vector of total expected occupations at each site.
"""
function calculate_site_occupations(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    up_occupation = zeros(Float64, params.L)
    dn_occupation = zeros(Float64, params.L)
    
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            idx = (i_up-1)*length(dn_basis) + i_dn
            prob = abs(ψ[idx])^2
            
            for site in 1:params.L
                if (up & (1 << (site-1))) != 0
                    up_occupation[site] += prob
                end
                if (dn & (1 << (site-1))) != 0
                    dn_occupation[site] += prob
                end
            end
        end
    end
    
    total_occupation = up_occupation + dn_occupation
    return up_occupation, dn_occupation, total_occupation
end

"""
    calculate_spin_correlation(ψ::Vector{ComplexF64}, params::HubbardParams)

Calculates the spin-spin correlation function <Sᵢᶻ Sⱼᶻ> between all pairs of sites (i, j)
from a given wavefunction `ψ`.

Args:
    ψ (Vector{ComplexF64}): The wavefunction vector in the combined basis.
    params (HubbardParams): Structure containing system parameters.

Returns:
    Matrix{Float64}: The L x L matrix containing the spin-spin correlations <Sᵢᶻ Sⱼᶻ>.
"""
function calculate_spin_correlation(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    spin_corr = zeros(Float64, params.L, params.L)
    
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            idx = (i_up-1)*length(dn_basis) + i_dn
            prob = abs(ψ[idx])^2
            
            for site_i in 1:params.L
                for site_j in 1:params.L
                    Sz_i = 0.5 * ((up & (1 << (site_i-1))) != 0 ? 1 : 0) - 
                           0.5 * ((dn & (1 << (site_i-1))) != 0 ? 1 : 0)
                    Sz_j = 0.5 * ((up & (1 << (site_j-1))) != 0 ? 1 : 0) - 
                           0.5 * ((dn & (1 << (site_j-1))) != 0 ? 1 : 0)
                    
                    spin_corr[site_i, site_j] += Sz_i * Sz_j * prob
                end
            end
        end
    end
    
    return spin_corr
end

"""
    plot_occupations(up_occ::Vector{Float64}, dn_occ::Vector{Float64}, total_occ::Vector{Float64})

Generates a plot showing the spin-up, spin-down, and total site occupations.

Args:
    up_occ (Vector{Float64}): Vector of spin-up occupations.
    dn_occ (Vector{Float64}): Vector of spin-down occupations.
    total_occ (Vector{Float64}): Vector of total occupations.

Returns:
    Plots.Plot: The generated plot object.
"""
function plot_occupations(up_occ, dn_occ, total_occ)
    p = plot(1:length(up_occ), up_occ, marker=:circle, label="Up", legend=:outertopright)
    plot!(p, 1:length(dn_occ), dn_occ, marker=:square, label="Down")
    plot!(p, 1:length(total_occ), total_occ, marker=:diamond, label="Total")
    xlabel!(p, "Site")
    ylabel!(p, "Occupation")
    title!(p, "Site Occupations")
    return p
end

"""
    plot_spin_correlation(spin_corr::Matrix{Float64})

Generates a heatmap plot of the spin-spin correlation matrix <Sᵢᶻ Sⱼᶻ>.

Args:
    spin_corr (Matrix{Float64}): The L x L spin-spin correlation matrix.

Returns:
    Plots.Plot: The generated heatmap plot object.
"""
function plot_spin_correlation(spin_corr)
    p = heatmap(spin_corr, aspect_ratio=1, c=:viridis)
    xlabel!(p, "Site i")
    ylabel!(p, "Site j")
    title!(p, "Spin-Spin Correlation Function <S_i^z S_j^z>")
    return p
end

"""
    calculate_double_occupancy(ψ::Vector{ComplexF64}, params::HubbardParams)

Calculates the expected double occupancy (probability of a site being occupied by
both an up and a down electron) for each site from a given wavefunction `ψ`.

Args:
    ψ (Vector{ComplexF64}): The wavefunction vector in the combined basis.
    params (HubbardParams): Structure containing system parameters.

Returns:
    Vector{Float64}: Vector containing the double occupancy for each site.
"""
function calculate_double_occupancy(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    double_occ = zeros(Float64, params.L)
    
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            idx = (i_up-1)*length(dn_basis) + i_dn
            prob = abs(ψ[idx])^2
            
            for site in 1:params.L
                if (up & (1 << (site-1))) != 0 && (dn & (1 << (site-1))) != 0
                    double_occ[site] += prob
                end
            end
        end
    end
    
    return double_occ
end

"""
    calculate_charge_gap(params::HubbardParams)

Calculates the charge gap for the Hubbard model at half-filling (N_up = N_dn = L/2)
using the definition Δ = E(N+1) + E(N-1) - 2*E(N), where N = N_up + N_dn.
Assumes the input `params` corresponds to the half-filled case (N).

Args:
    params (HubbardParams): Structure containing system parameters for the half-filled system.

Returns:
    Float64: The calculated charge gap.
"""
function calculate_charge_gap(params)
    H_half = build_hamiltonian(params)
    E_half = eigvals(Symmetric(H_half))[1]
    
    params_plus = HubbardParams(params.L, params.N_up + 1, params.N_dn, params.t, params.U)
    H_plus = build_hamiltonian(params_plus)
    E_plus = eigvals(Symmetric(H_plus))[1]
    
    params_minus = HubbardParams(params.L, params.N_up - 1, params.N_dn, params.t, params.U)
    H_minus = build_hamiltonian(params_minus)
    E_minus = eigvals(Symmetric(H_minus))[1]
    
    return E_plus + E_minus - 2*E_half
end

"""
    analytical_charge_gap(U::Float64, t::Float64, L::Int)

Provides an analytical approximation for the charge gap, primarily valid in the large U/t limit.

Args:
    U (Float64): On-site interaction strength.
    t (Float64): Hopping strength.
    L (Int): Number of sites (used for finite-size correction).

Returns:
    Float64: Approximate analytical charge gap.
"""
function analytical_charge_gap(U, t, L)
    if U/t > 4
        return U - 4*t^2/U * (1 - 1/L)
    else
        return max(0, U - 4*t)
    end
end

"""
    analytical_double_occupancy(U::Float64, t::Float64)

Provides an analytical approximation for the average double occupancy per site at half-filling.
Valid for U=0 and gives approximate scaling for large U.

Args:
    U (Float64): On-site interaction strength.
    t (Float64): Hopping strength.

Returns:
    Float64: Approximate analytical double occupancy.
"""
function analytical_double_occupancy(U, t)
    if U == 0
        return 0.25  # Uncorrelated limit
    else
        return 0.25 / (1 + U/(4*t))
    end
end

"""
    calculate_critical_point(U_t_values::Vector{Float64}, charge_gaps::Vector{Float64})

Estimates the critical U/t for the Mott transition by finding the maximum
of the numerical derivative of the charge gap with respect to U/t.

Args:
    U_t_values (Vector{Float64}): Vector of U/t ratios used in calculations.
    charge_gaps (Vector{Float64}): Corresponding calculated charge gaps.

Returns:
    Tuple{Float64, Vector{Float64}}: A tuple containing:
        - `critical_U_t`: Estimated critical U/t value.
        - `derivatives`: Vector of calculated numerical derivatives d(Gap)/d(U/t).
"""
function calculate_critical_point(U_t_values, charge_gaps)
    derivatives = Float64[]
    for i in 2:length(U_t_values)
        dg_du = (charge_gaps[i] - charge_gaps[i-1]) / (U_t_values[i] - U_t_values[i-1])
        push!(derivatives, dg_du)
    end
    
    max_deriv, idx = findmax(derivatives)
    critical_U_t = (U_t_values[idx] + U_t_values[idx+1]) / 2
    
    return critical_U_t, derivatives
end

"""
    plot_mott_transition(U_t_values, charge_gaps, double_occs, validation_results)

Generates a combined plot showing the numerical and analytical charge gap,
double occupancy, and the derivative of the charge gap as a function of U/t
to visualize the Mott transition.

Args:
    U_t_values: Vector of U/t ratios.
    charge_gaps: Vector of numerical charge gaps.
    double_occs: Vector of numerical average double occupancies.
    validation_results: A structure or tuple containing analytical gaps, analytical occupancies,
                        the estimated critical U/t, and the derivatives.

Returns:
    Plots.Plot: The combined plot object.
"""
function plot_mott_transition(U_t_values, charge_gaps, double_occs, validation_results)
    p1 = plot(U_t_values, charge_gaps, 
        label="Numerical", 
        marker=:circle, 
        linewidth=2,
        xlabel="U/t",
        ylabel="Charge Gap",
        title="Mott Transition: Charge Gap"
    )
    
    plot!(p1, U_t_values, validation_results.analytical_gaps, 
        label="Analytical", 
        linestyle=:dash, 
        linewidth=2
    )
    
    vline!(p1, [validation_results.critical_U_t], 
        label="Critical U/t ≈ $(round(validation_results.critical_U_t, digits=2))", 
        linestyle=:dot, 
        linewidth=2
    )

    p2 = plot(U_t_values, double_occs,
        label="Numerical", 
        marker=:square, 
        linewidth=2,
        color=:red,
        xlabel="U/t",
        ylabel="Average Double Occupancy",
        title="Mott Transition: Double Occupancy"
    )
    
    plot!(p2, U_t_values, validation_results.analytical_occs, 
        label="Analytical", 
        linestyle=:dash, 
        linewidth=2,
        color=:darkred
    )
    
    vline!(p2, [validation_results.critical_U_t], 
        label="Critical U/t", 
        linestyle=:dot, 
        linewidth=2
    )

    p3 = plot(U_t_values[2:end], validation_results.derivatives,
        label="dΔ/d(U/t)", 
        linewidth=2,
        color=:purple,
        xlabel="U/t",
        ylabel="dΔ/d(U/t)",
        title="Charge Gap Derivative"
    )
    
    vline!(p3, [validation_results.critical_U_t], 
        label="Critical U/t", 
        linestyle=:dot, 
        linewidth=2
    )

    return plot(p1, p2, p3, layout=(3,1), size=(800, 900), legend=true)
end

end # module
