using LinearAlgebra      # norm, Hermitian
using Plots              # plotting backend
using Statistics         # basic stats

# Load the original modules
include("src/hubbardinit.jl")
include("src/reciprocal.jl")
include("src/meanfield.jl")

using .MeanField          # bring mean-field functions into scope

# 1) Parameters
U    = 1.0                  # set U=1 so that t/U is just t
ne   = 0.8                  # electron filling per site
Nk   = 100                  # k‐point grid density (e.g. 100×100)
β    = 250                 # inverse temperature (INCREASED significantly)
tol  = 1e-5                 # SCF convergence tolerance
maxit= 200                  # max SCF iterations
δ    = 0.1                  # AFM “seed” imbalance

# 2) Range of t/U values to scan
t_over_U = range(0.10, 0.25, length=30)
num_points = length(t_over_U)

# 3) Storage for the two curves - Ensure re-initialization
E_FM  = Float64[]
sizehint!(E_FM, num_points) # Optional: pre-allocate memory
E_AFM = Float64[] # Initialized empty
sizehint!(E_AFM, num_points) # Optional: pre-allocate memory # Initialized empty

# 4) Loop over t/U, solve SCF in FM and AFM trials
println("Starting SCF calculations for $(num_points) points...")
for (idx, t) in enumerate(t_over_U)
    p = MFParams(U=U, t=t, ne=ne, Nk=Nk, β=β, tol=tol, maxiter=maxit, mixing_alpha=0.5) # Ensure mixing_alpha is here
    println("  Calculating t/U = $t (Point $idx/$num_points)")

    # --- Ferromagnetic trial ---
    local Efm = NaN
    local nup_fm_conv = [NaN, NaN] # Store converged densities
    local ndown_fm_conv = [NaN, NaN]
    init_FM = (fill(ne,2), zeros(2))
    try
        nup_fm_conv, ndown_fm_conv, Efm_calc = self_consistent_mf(p; init=init_FM) # Capture densities
        if isfinite(Efm_calc)
            Efm = Efm_calc
            # --- Print Converged Densities ---
            println("    FM Converged: E=$Efm, n↑=", nup_fm_conv, ", n↓=", ndown_fm_conv)
            # --- End Print ---
        else
            println("    Warning: FM energy calculation resulted in non-finite value ($Efm_calc)")
        end
    catch e
        println("    Error during FM calculation at t/U = $t: $e")
    end
    push!(E_FM, Efm)

    # --- Antiferromagnetic trial ---
    local Eafm = NaN
    local nup_afm_conv = [NaN, NaN] # Store converged densities
    local ndown_afm_conv = [NaN, NaN]
    init_AFM = ([ne/2+δ, ne/2-δ], [ne/2-δ, ne/2+δ])
    try
        nup_afm_conv, ndown_afm_conv, Eafm_calc = self_consistent_mf(p; init=init_AFM) # Capture densities
         if isfinite(Eafm_calc)
            Eafm = Eafm_calc
            # --- Print Converged Densities ---
            println("    AFM Converged: E=$Eafm, n↑=", nup_afm_conv, ", n↓=", ndown_afm_conv)
            # --- End Print ---
        else
            println("    Warning: AFM energy calculation resulted in non-finite value ($Eafm_calc)")
        end
    catch e
        println("    Error during AFM calculation at t/U = $t: $e")
    end
    push!(E_AFM, Eafm)
end
println("SCF calculations finished.")

# --- Add Assertions and Checks ---
println("Length of t_over_U: ", length(t_over_U))
println("Length of E_FM: ", length(E_FM))
println("Length of E_AFM: ", length(E_AFM))

@assert length(E_FM) == num_points "Length mismatch for E_FM! Expected $num_points, got $(length(E_FM))"
@assert length(E_AFM) == num_points "Length mismatch for E_AFM! Expected $num_points, got $(length(E_AFM))"
# --- End Assertions ---

# Filter out NaN values before plotting
fm_finite_mask = isfinite.(E_FM)
afm_finite_mask = isfinite.(E_AFM)

t_fm = t_over_U[fm_finite_mask]
E_fm_finite = E_FM[fm_finite_mask]
t_afm = t_over_U[afm_finite_mask]
E_afm_finite = E_AFM[afm_finite_mask]

println("Plotting $(length(E_afm_finite)) finite AFM points and $(length(E_fm_finite)) finite FM points.")

# 5) Make the plot using only finite values
if isempty(E_afm_finite) && isempty(E_fm_finite)
    println("Error: No finite energy values to plot.")
    plt = plot() # Create an empty plot
else
    if !isempty(E_afm_finite)
        plt = plot(t_afm, E_afm_finite, label="AFM", marker=:circle, lw=2)
    else
        plt = plot()
        println("Warning: No finite AFM points to plot.")
    end
    if !isempty(E_fm_finite)
        plot!(plt, t_fm, E_fm_finite, label="FM", marker=:square, lw=2)
    else
         println("Warning: No finite FM points to plot.")
    end
end

xlabel!("t / U")
ylabel!("Free energy per cell (in t units)") # Note: Energy, not necessarily Free Energy if beta is high
title!("Magnetic Energy vs t/U at n_e = $ne, β = $β") # Updated title
#legend!(:bottomleft) # Uncomment if needed

# 6) Save or display
savefig("figure4.png")
display(plt)

