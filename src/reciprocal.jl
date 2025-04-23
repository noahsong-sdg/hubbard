module ReciprocalSpace

using LinearAlgebra
using SparseArrays
using Plots
using Statistics
using Revise
using ..HInit # Needed for HubbardParams in calculate_dispersion

export calculate_dispersion, 
       generate_k_path, 
       plot_dispersion, 
       plot_bands,
       gamma_k,
       get_bands,
       # Export the constants
       a, b1, b2, k,
       t, Γ, X, M, KPATH,
       plot_band_structure

# Constants related to reciprocal space
const a = 1.0  # Lattice constant
const b1 = [1.0, 1.0]
const b2 = [1.0, -1.0]
const k = [0, 0]  # Default k-point
const t = 1.0  # Default hopping parameter

const Γ = [0.0, 0.0]
const X = [π, 0.0]
const M = [π, π]
const KPATH = [Γ, X, M, Γ]

# Function to calculate dispersion relation in Brillouin zone
function calculate_dispersion(params::HubbardParams, k_points)
    # For non-interacting case (U=0), we can use the tight-binding dispersion
    t = params.t
    L = params.L
    
    # Assuming 2D square lattice with L = L_x × L_y
    L_x = Int(sqrt(L))
    L_y = L ÷ L_x
    
    # Calculate dispersion along the path
    energies = []
    for (kx, ky) in k_points
        # Tight-binding dispersion in 2D: -2t(cos(kx) + cos(ky))
        ε_k = -2*t*(cos(kx) + cos(ky))
        push!(energies, ε_k)
    end
    
    return energies
end

# Generate path through Brillouin zone
function generate_k_path(points, num_points=100)
    path = []
    positions = [0.0]

    # Iterate using axes, excluding the last point
    for i in axes(points, 1)[begin:end-1]
        p1 = points[i]
        p2 = points[i+1]

        # Generate points along the line using LinRange
        # Note: LinRange includes endpoints, adjust if duplicates are unwanted
        # Here, we likely want num_points intervals, so num_points+1 points
        for t in LinRange(0, 1, num_points) # Use LinRange
            kx = p1[1] * (1-t) + p2[1] * t
            ky = p1[2] * (1-t) + p2[2] * t
            # Avoid adding duplicate points if t=1 overlaps with next segment's t=0
            # A simple fix is to exclude the last point (t=1) except for the final segment
            if t < 1.0 || i == lastindex(points) - 1
                 push!(path, (kx, ky))
            end
        end

        # Add position marker for the next high symmetry point
        push!(positions, positions[end] + norm(p2 .- p1)) # Use norm
    end
    # Ensure the very last point is included if generate_k_path is expected to return it
    if !isempty(points) && (isempty(path) || path[end] != points[end])
         push!(path, points[end])
    end

    return path, positions
end

gamma_k(kvec, b1, b2) = -(1 + exp(-im * dot(kvec, b1)) + exp(-im * dot(kvec, b2)) + exp(-im * dot(kvec, b1 + b2)))

function get_bands(kvec, b1, b2)
    gamk = gamma_k(kvec, b1, b2)
    H = [0 gamk; conj(gamk) 0]
    e = eigen(H)
    return e.values
end

function plot_band_structure(ksteps, PLOT = false)
    Γ = [0.0, 0.0]
    X = [π, 0.0]
    M = [π, π]
    path = [Γ, X, M, Γ]

    # Interpolate k-points
    kvec = []
    # Iterate using axes, excluding the last point
    for i in axes(path, 1)[begin:end-1]
        # Use LinRange, exclude endpoint t=1 except for the last segment
        for t in LinRange(0, 1, ksteps)[1:end-1]
            push!(kvec, path[i] * (1 - t) + path[i+1] * t)
        end
    end
    # Add the final point of the path
     push!(kvec, path[end])

    εk_minus = Float64[]
    εk_plus = Float64[]
    # Use eachindex for iterating over kvec
    for k_idx in eachindex(kvec)
        k = kvec[k_idx]
        em, ep = get_bands(k, b1, b2)
        push!(εk_minus, em)
        push!(εk_plus, ep)
    end
    if PLOT 
        plot(εk_minus, label="ε₋", linewidth=2)
        plot!(εk_plus, label="ε₊", linewidth=2)
    end
    return εk_minus, εk_plus
end





end
