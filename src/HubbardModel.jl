module HubbardModel

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf
using Statistics

# Include files defining submodules *inside* this module
include("hubbardinit.jl") 
include("meanfield.jl")
include("reciprocal.jl")
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
       calculate_dispersion, # Exported from HInit or ReciprocalSpace? Ensure it's exported from the correct one.
       
       # Exports from ReciprocalSpace
       generate_k_path, 
       plot_dispersion, 
       gamma_k,
       calculate_bands,
       a, b1, b2, k, t, Γ, X, M, KPATH, # Export constants
       KPATH_B,
       plot_bands, calculate_dos, plot_dos, calculate_fermi_level,

       # Exports from MeanField
       MFParams, self_consistent_mf, compute_phase_diagram, mean_field_hamiltonian

end
