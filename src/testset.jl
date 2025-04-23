module testset

#########################################################
################### Mott Transition stuff#####################################
#########################################################
function debug_hamiltonian(params::HubbardParams)
    up_basis = create_basis(params.L, params.N_up)
    dn_basis = create_basis(params.L, params.N_dn)
    
    println("Parameters: L=$(params.L), N_up=$(params.N_up), N_dn=$(params.N_dn), t=$(params.t), U=$(params.U)")
    println("Up basis ($(length(up_basis)) states):")
    for (i, state) in enumerate(up_basis)
        println("  $i: $(string(state, base=2, pad=params.L)) (decimal: $state)")
    end
    
    println("\nDown basis ($(length(dn_basis)) states):")
    for (i, state) in enumerate(dn_basis)
        println("  $i: $(string(state, base=2, pad=params.L)) (decimal: $state)")
    end
    
    # Track if any terms are added
    hopping_found = false
    interaction_found = false
    
    # Build Hamiltonian with logging
    dim = length(up_basis) * length(dn_basis)
    H = zeros(Float64, dim, dim)
    
    for (i_up, up) in enumerate(up_basis)
        up_bin = string(up, base=2, pad=params.L)
        for (i_dn, dn) in enumerate(dn_basis)
            dn_bin = string(dn, base=2, pad=params.L)
            idx1 = (i_up-1)*length(dn_basis) + i_dn
            
            # Check hopping
            for site in 1:params.L
                next_site = site % params.L + 1
                
                # Try up spin hopping
                if (up & (1 << (site-1))) != 0 && (up & (1 << (next_site-1))) == 0
                    new_up = up ⊻ (1 << (site-1)) ⊻ (1 << (next_site-1))
                    j_up = findfirst(==(new_up), up_basis)
                    if j_up !== nothing
                        idx2 = (j_up-1)*length(dn_basis) + i_dn
                        H[idx1, idx2] -= params.t
                        H[idx2, idx1] -= params.t
                        hopping_found = true
                        println("UP HOPPING: $(up_bin) -> $(string(new_up, base=2, pad=params.L)), site $site to $next_site")
                    end
                end
                
                # Try down spin hopping
                if (dn & (1 << (site-1))) != 0 && (dn & (1 << (next_site-1))) == 0
                    new_dn = dn ⊻ (1 << (site-1)) ⊻ (1 << (next_site-1))
                    j_dn = findfirst(==(new_dn), dn_basis)
                    if j_dn !== nothing
                        idx2 = (i_up-1)*length(dn_basis) + j_dn
                        H[idx1, idx2] -= params.t
                        H[idx2, idx1] -= params.t
                        hopping_found = true
                        println("DOWN HOPPING: $(dn_bin) -> $(string(new_dn, base=2, pad=params.L)), site $site to $next_site")
                    end
                end
            end
            
            # Check interaction
            for site in 1:params.L
                if (up & (1 << (site-1))) != 0 && (dn & (1 << (site-1))) != 0
                    H[idx1, idx1] += params.U
                    interaction_found = true
                    println("INTERACTION: Up=$(up_bin), Down=$(dn_bin), site $site")
                end
            end
        end
    end
    
    # Print summary
    println("\nSUMMARY:")
    println("Hopping terms added: $(hopping_found ? "YES" : "NO")")
    println("Interaction terms added: $(interaction_found ? "YES" : "NO")")
    println("Non-zero elements in H: $(count(!iszero, H)) of $(dim*dim)")
    
    return H
end
# Function to validate Mott transition results
function validate_mott_transition(U_t_values, charge_gaps, double_occs, L)
    t = 1.0  # Using t=1 as the energy unit
    
    # Calculate analytical predictions
    analytical_gaps = [analytical_charge_gap(U*t, t, L) for U in U_t_values]
    analytical_occs = [analytical_double_occupancy(U*t, t) for U in U_t_values]
    
    # Calculate critical U/t
    critical_U_t, derivatives = calculate_critical_point(U_t_values, charge_gaps)
    
    # Calculate numerical errors
    gap_errors = abs.(charge_gaps - analytical_gaps) ./ analytical_gaps
    occ_errors = abs.(double_occs - analytical_occs) ./ analytical_occs
    
    # Mean relative errors
    mean_gap_error = mean(filter(!isnan, gap_errors))
    mean_occ_error = mean(filter(!isnan, occ_errors))
    
    # Behavior checks
    monotonic_gap = all(diff(charge_gaps) .>= 0)
    monotonic_occ = all(diff(double_occs) .<= 0)
    
    # Check limiting behaviors
    small_U_gap = charge_gaps[1] < 2.0  # Should be small for small U/t
    large_U_gap_linear = isapprox(charge_gaps[end]/U_t_values[end], 1.0, atol=0.3)  # Should approach U for large U/t
    small_U_occ = isapprox(double_occs[1], 0.25, atol=0.05)  # Should be close to 0.25 for small U/t
    large_U_occ_small = double_occs[end] < 0.05  # Should be close to 0 for large U/t
    
    # Return validation results
    return (
        critical_U_t=critical_U_t,
        derivatives=derivatives,
        analytical_gaps=analytical_gaps,
        analytical_occs=analytical_occs,
        monotonic_gap=monotonic_gap,
        monotonic_occ=monotonic_occ,
        small_U_gap=small_U_gap,
        large_U_gap_linear=large_U_gap_linear,
        small_U_occ=small_U_occ,
        large_U_occ_small=large_U_occ_small,
        mean_gap_error=mean_gap_error,
        mean_occ_error=mean_occ_error
    )
end


end
