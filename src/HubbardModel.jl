module HubbardModel

using LinearAlgebra
using SparseArrays
using Plots
using Statistics 
using Revise 

# Include files defining submodules *inside* this module
# Assuming hubbardinit.jl and reciprocal.jl are in the same directory as HubbardModel.jl (e.g., src/)
include("hubbardinit.jl") 
include("reciprocal.jl")
include("meanfield.jl")
# Use the submodules defined by the included files
using .HInit
using .ReciprocalSpace
using .MeanField
# Do NOT re-export HInit and ReciprocalSpace
# export HInit # Remove
# export ReciprocalSpace # Remove

export HubbardParams, # Export types/functions from submodules if needed at HubbardModel level
       create_basis, 
       build_hamiltonian, 
       calculate_site_occupations, 
       calculate_spin_correlation,
       plot_occupations, 
       plot_spin_correlation,
       calculate_double_occupancy,
       calculate_charge_gap,
       # ... other exports from HInit ...
       calculate_dispersion, # Exported from HInit or ReciprocalSpace? Ensure it's exported from the correct one.
       
       # Exports from ReciprocalSpace
       generate_k_path, 
       plot_dispersion, 
       plot_bands,
       gamma_k,
       get_bands,
       a, b1, b2, k, t, Γ, X, M, KPATH, # Export constants
       plot_band_structure,
       MFParams, self_consistent_mf, compute_phase_diagram, mean_field_hamiltonian
       # Note: ksteps was missing from export list, add if needed

end
