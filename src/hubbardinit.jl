module HInit

using LinearAlgebra
using SparseArrays
using Plots
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
         calculate_dispersion
# System parameters
struct HubbardParams
    L::Int      # Number of sites
    N_up::Int   # Number of up-spin electrons
    N_dn::Int   # Number of down-spin electrons
    t::Float64  # Hopping strength
    U::Float64  # On-site interaction
end

function create_basis(L, N)
    # Generate all possible configurations of N electrons on L sites
    if N > L || N < 0
        return Int[]  # Invalid parameters
    end
    
    # Calculate the number of states - binomial coefficient (L choose N)
    num_states = binomial(L, N)
    basis = Vector{Int}(undef, num_states)
    
    if N == 0
        basis[1] = 0  # No electrons means all zeros
        return basis
    end
    
    # Start with the smallest valid state: N ones at the right
    state = (1 << N) - 1
    idx = 1
    basis[idx] = state
    
    # Generate all other states using Gosper's hack
    while idx < num_states
        # Gosper's hack for next combination
        x = state & -state
        y = state + x
        state = (((state & ~y) ÷ x) >> 1) | y
        
        idx += 1
        basis[idx] = state
    end
    
    return basis
end



function build_hamiltonian(params::HubbardParams)
    # Create basis for up and down spins
    # generates all possible configurations of N electrons on L sites
    # does so in binary representation: 0b1010 means electrons on sites 1 and 3
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    # the hilbert space simension is the product of upspin and downspin basis sizes 
    dim = length(up_basis) * length(dn_basis)
    H = zeros(Float64, dim, dim)
    
    # Add hopping terms (Kinetic Energy)
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            for site in 1:params.L
                # implements periodic boundary conditions - the last site is connected to the first
                next_site = site % params.L + 1
                
                # Hop up spins -  move electron from 'site' to 'next_site'
                # idx1 is the index of the initial state in the full H
                # idx2 is the index of the final state in the full H
                # the hopping term is added to the off-diagonal elements of the Hamiltonian
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
                
                # Hop down spins
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
            
            # Add interaction term
            # if a site is doubly occupied, add the interaction term to the diagonal
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

# Calculate the site occupations from the ground state wavefunction
function calculate_site_occupations(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    # Initialize arrays to store up, down and total occupations
    up_occupation = zeros(Float64, params.L)
    dn_occupation = zeros(Float64, params.L)
    
    # Calculate occupations
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            idx = (i_up-1)*length(dn_basis) + i_dn
            prob = abs(ψ[idx])^2
            
            # Check each site
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

# Calculate spin-spin correlation function
function calculate_spin_correlation(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    # Initialize correlation matrix
    spin_corr = zeros(Float64, params.L, params.L)
    
    for (i_up, up) in enumerate(up_basis)
        for (i_dn, dn) in enumerate(dn_basis)
            idx = (i_up-1)*length(dn_basis) + i_dn
            prob = abs(ψ[idx])^2
            
            for site_i in 1:params.L
                for site_j in 1:params.L
                    # Calculate S_z at site i and site j
                    Sz_i = 0.5 * ((up & (1 << (site_i-1))) != 0 ? 1 : 0) - 
                           0.5 * ((dn & (1 << (site_i-1))) != 0 ? 1 : 0)
                    Sz_j = 0.5 * ((up & (1 << (site_j-1))) != 0 ? 1 : 0) - 
                           0.5 * ((dn & (1 << (site_j-1))) != 0 ? 1 : 0)
                    
                    # Add contribution to correlation
                    spin_corr[site_i, site_j] += Sz_i * Sz_j * prob
                end
            end
        end
    end
    
    return spin_corr
end

# Plot site occupations
function plot_occupations(up_occ, dn_occ, total_occ)
    p = plot(1:length(up_occ), up_occ, marker=:circle, label="Up", legend=:outertopright)
    plot!(p, 1:length(dn_occ), dn_occ, marker=:square, label="Down")
    plot!(p, 1:length(total_occ), total_occ, marker=:diamond, label="Total")
    xlabel!(p, "Site")
    ylabel!(p, "Occupation")
    title!(p, "Site Occupations")
    return p
end

# Plot spin correlations
function plot_spin_correlation(spin_corr)
    p = heatmap(spin_corr, aspect_ratio=1, c=:viridis)
    xlabel!(p, "Site i")
    ylabel!(p, "Site j")
    title!(p, "Spin-Spin Correlation Function <S_i^z S_j^z>")
    return p
end

# Function to calculate double occupancy
function calculate_double_occupancy(ψ, params)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    # Initialize array to store double occupancy
    double_occ = zeros(Float64, params.L)
    
    # Calculate double occupancy at each site
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

# Function to calculate the charge gap
function calculate_charge_gap(params)
    # Ground state energy at half-filling
    H_half = build_hamiltonian(params)
    E_half = eigvals(Symmetric(H_half))[1]
    
    # Ground state energy with one extra electron
    params_plus = HubbardParams(params.L, params.N_up + 1, params.N_dn, params.t, params.U)
    H_plus = build_hamiltonian(params_plus)
    E_plus = eigvals(Symmetric(H_plus))[1]
    
    # Ground state energy with one fewer electron
    params_minus = HubbardParams(params.L, params.N_up - 1, params.N_dn, params.t, params.U)
    H_minus = build_hamiltonian(params_minus)
    E_minus = eigvals(Symmetric(H_minus))[1]
    
    # The charge gap is approximately E(N+1) + E(N-1) - 2*E(N)
    return E_plus + E_minus - 2*E_half
end

# Analytical approximation of charge gap for large U/t
function analytical_charge_gap(U, t, L)
    # For large U/t, the charge gap approaches U
    # With finite-size corrections that scale as t²/U
    if U/t > 4
        return U - 4*t^2/U * (1 - 1/L)
    else
        # For small U, use a simple approximation based on mean-field theory
        # This is a rough approximation for illustration purposes
        return max(0, U - 4*t)
    end
end

# Analytical approximation for double occupancy
function analytical_double_occupancy(U, t)
    # For U=0, double occupancy is 0.25 at half-filling
    # For large U, it scales approximately as t/U
    if U == 0
        return 0.25  # Uncorrelated limit
    else
        # Approximate form that transitions from 0.25 to t/U scaling
        return 0.25 / (1 + U/(4*t))
    end
end

# Calculate critical point by looking at the derivative of the charge gap
function calculate_critical_point(U_t_values, charge_gaps)
    # Calculate numerical derivative
    derivatives = Float64[]
    for i in 2:length(U_t_values)
        dg_du = (charge_gaps[i] - charge_gaps[i-1]) / (U_t_values[i] - U_t_values[i-1])
        push!(derivatives, dg_du)
    end
    
    # Find the point of maximum slope
    max_deriv, idx = findmax(derivatives)
    critical_U_t = (U_t_values[idx] + U_t_values[idx+1]) / 2
    
    return critical_U_t, derivatives
end












##############################################################33
################ Mott Stuff ##########################
###############################################################


# Enhanced plotting for Mott transition analysis
function plot_mott_transition(U_t_values, charge_gaps, double_occs, validation_results)
    # Create the charge gap plot
    p1 = plot(U_t_values, charge_gaps, 
        label="Numerical", 
        marker=:circle, 
        linewidth=2,
        xlabel="U/t",
        ylabel="Charge Gap",
        title="Mott Transition: Charge Gap"
    )
    
    # Add analytical prediction for comparison
    plot!(p1, U_t_values, validation_results.analytical_gaps, 
        label="Analytical", 
        linestyle=:dash, 
        linewidth=2
    )
    
    # Add reference line at critical point
    vline!(p1, [validation_results.critical_U_t], 
        label="Critical U/t ≈ $(round(validation_results.critical_U_t, digits=2))", 
        linestyle=:dot, 
        linewidth=2
    )

    # Create the double occupancy plot
    p2 = plot(U_t_values, double_occs,
        label="Numerical", 
        marker=:square, 
        linewidth=2,
        color=:red,
        xlabel="U/t",
        ylabel="Average Double Occupancy",
        title="Mott Transition: Double Occupancy"
    )
    
    # Add analytical prediction for comparison
    plot!(p2, U_t_values, validation_results.analytical_occs, 
        label="Analytical", 
        linestyle=:dash, 
        linewidth=2,
        color=:darkred
    )
    
    # Add reference line at critical point
    vline!(p2, [validation_results.critical_U_t], 
        label="Critical U/t", 
        linestyle=:dot, 
        linewidth=2
    )

    # Create derivative plot to show the critical point more clearly
    p3 = plot(U_t_values[2:end], validation_results.derivatives,
        label="dΔ/d(U/t)", 
        linewidth=2,
        color=:purple,
        xlabel="U/t",
        ylabel="dΔ/d(U/t)",
        title="Charge Gap Derivative"
    )
    
    # Add reference line at critical point
    vline!(p3, [validation_results.critical_U_t], 
        label="Critical U/t", 
        linestyle=:dot, 
        linewidth=2
    )

    # Combine all plots
    return plot(p1, p2, p3, layout=(3,1), size=(800, 900), legend=true)
end

end # module
