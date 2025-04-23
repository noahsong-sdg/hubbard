module MottInsulator


using LinearAlgebra
using SparseArrays
using Plots
using Statistics 
using Revise

includet("HubbardModel.jl")
using .HubbardModel

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
       calculate_critical_point

end
