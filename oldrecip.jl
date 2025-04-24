module ReciprocalSpace

using LinearAlgebra
using SparseArrays
using Plots
using Statistics
using Revise
using Printf
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
"""
    generate_k_path(points, num_points_per_segment=50)

Generates a list of k-points interpolating linearly between the given `points`.
Also calculates the cumulative distance along the path.

Args:
    points: A list of k-space coordinates (e.g., [[0,0], [π,0], [π,π], [0,0]]).
    num_points_per_segment: Number of points to generate for each segment between consecutive points in `points`.

Returns:
    k_path: A vector of k-point coordinates (tuples).
    k_dist: A vector of cumulative distances corresponding to each point in `k_path`.
    tick_positions: A vector of distances corresponding to the input high-symmetry `points`.
    tick_labels: A vector of labels for the high-symmetry points (currently hardcoded for Γ, X, M).
"""
function generate_k_path(points, num_points_per_segment=50)
    k_path = []
    k_dist = [0.0] # Start distance at 0
    tick_positions = [0.0]
    # Simple labeling for now, could be made more general
    point_labels = ["Γ", "X", "M", "Γ", "Y", "Z"] # Extend as needed
    tick_labels = [point_labels[i] for i in 1:length(points)]

    current_dist = 0.0
    # Iterate through segments defined by consecutive points
    for i in 1:(length(points)-1)
        p1 = points[i]
        p2 = points[i+1]
        segment_vec = p2 .- p1
        segment_len = norm(segment_vec)

        # Generate points along the segment using LinRange
        # Generate num_points_per_segment intervals, so num_points_per_segment+1 points
        # Exclude the last point (t=1.0) to avoid duplication with the start of the next segment
        # unless it's the very last segment
        num_steps = (i == length(points) - 1) ? num_points_per_segment : num_points_per_segment -1
        for step in 0:num_steps
            t = step / num_points_per_segment
            k = p1 .+ t .* segment_vec
            # Add the first point of the first segment, or points from subsequent steps
            if i == 1 && step == 0
                push!(k_path, tuple(k...))
            elseif step > 0
                 push!(k_path, tuple(k...))
                 # Calculate distance from the previous point in the path
                 dist_step = norm(k .- k_path[end-1])
                 current_dist += dist_step
                 push!(k_dist, current_dist)
            end
        end
        # Store the position of the high-symmetry point at the end of the segment
        push!(tick_positions, current_dist)
    end

    # Ensure k_dist has the same length as k_path
    if length(k_dist) != length(k_path)
       println("Warning: k_dist and k_path length mismatch. Recalculating k_dist.")
       k_dist = [0.0]
       for idx = 2:length(k_path)
           push!(k_dist, k_dist[end] + norm(k_path[idx] .- k_path[idx-1]))
       end
    end

    return k_path, k_dist, tick_positions, tick_labels
end

gamma_k(kvec, b1, b2) = -(1 + exp(-im * dot(kvec, b1)) + exp(-im * dot(kvec, b2)) + exp(-im * dot(kvec, b1 + b2)))

function get_bands(kvec, b1, b2)
    gamk = gamma_k(kvec, b1, b2)
    H = [0 gamk; conj(gamk) 0]
    e = eigen(H)
    return e.values
end

# This function is less general now, generate_k_path is preferred
# Kept for potential backward compatibility or specific use cases if needed,
# but consider removing or refactoring if unused.
function plot_band_structure(ksteps, PLOT = false)
    # Use the more general generate_k_path
    kvec, kdist, tick_positions, tick_labels = generate_k_path(KPATH, ksteps)

    εk_minus = Float64[]
    εk_plus = Float64[]
    for k in kvec
        em, ep = get_bands(k, b1, b2)
        push!(εk_minus, em)
        push!(εk_plus, ep)
    end
    if PLOT 
        p = plot(kdist, εk_minus, label="ε₋", linewidth=2)
        plot!(p, kdist, εk_plus, label="ε₊", linewidth=2)
        xticks!(p, tick_positions, tick_labels)
        xlabel!(p, "k-path")
        ylabel!(p, "Energy")
        title!(p, "Band Structure")
        display(p)
    end
    return εk_minus, εk_plus
end

end
