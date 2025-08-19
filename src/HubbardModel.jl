module HubbardModel

using LinearAlgebra 
using SparseArrays
using StatsBase
using Plots
using Statistics
using Revise
using Printf
using Parameters

# Include all files directly into the main module namespace
# This eliminates submodule conflicts and type confusion
include("hubbardinit.jl") 
include("meanfield.jl")
include("reciprocal.jl")

# All types and functions are now available directly in HubbardModel namespace
# No need for explicit exports since everything is in the same namespace

end
