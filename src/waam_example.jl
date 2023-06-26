include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: PlanarWAAMHardness, AdditiveProblem, QuadraticObjective, optimize_trajectory, generate_wall_z₀, animate_state_history, input_idle, PlanarGMAWDynamics, animate_measurement_history, constraints!, PlanarWAAM, PlanarWAAMPrescribedMotion, state_min, property_min, temperature!
using Plots

A = 1e4
τ = 9625

γᵣ = 5
γₕ = 10
bₕ = 2
wₓ = 2.5

σ = 5.670374419 * 10^(-8) # Stefan-Boltzmann, W / (m⁴⋅ K⁴)

h∞ = 10 # W / m^2 K
h₀ = 7500 # W / m^2 K
hₐᵣ = 500
η = 0.95

kₘ = 34 # W / mK
kₐ = 4.6e-2 # W / mK

ρₘ = 7826.0e6 # kg / m^3 # mg
ρₐ = 1.293e6 # kg / m^3 # mg

cₚₘ = 502.416e-6 # J / kg K # mg 
cₚₐ = 400e-6#717.0e-6 # J / kg K # mg 

T∞ = 295.0 # K
T₀ = 295.0 # K
wire_diam = 0.001143 # m, aka 0.045in
Tₗ = 1784.0 # K, liquidus

nrows = 4
ncols = 16
nthick = 7
nvox = nrows * ncols

Tmin = T₀
Tmax = 1900 # K

l = 0.001 # m, aka 1mm
w = 0.010 # m, aka 10mm

# process = PlanarWAAMHardness(nrows, ncols, nthick, l, k, ρ, cₚ, T∞, T₀, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, A, τ)
# process = PlanarWAAM(nrows, ncols, nthick, l, k, ρ, cₚ, T∞, T₀, Tₗ, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ)
process = PlanarWAAMPrescribedMotion(nrows, ncols, l, w, kₘ, kₐ, ρₘ, ρₐ, cₚₘ, cₚₐ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, Tmin, Tmax, l, l * (ncols - 1), 0.0, 1.8)


# Q = Diagonal([1e-4 * ones(nvox); 1e-3 * ones(nvox); 1e0 * ones(nvox); 1e-2 * ones(4)]) #1e4 mass in kg
# Q = Diagonal([1e-4 * ones(nvox); 1e-3 * ones(nvox); 1e-2 * ones(4)]) #1e4 mass in kg
Q = Diagonal([1e-5 * ones(nvox); 1e0 * ones(nvox); 1e-2 * ones(1)]) #1e4 mass in kg
# R = Diagonal(1e-2 * ones(4))
R = Diagonal(1e-2 * ones(1))
Qf = 10 * Q #cₚ * T∞ * 1e-40 * ones(nvox)
# x₀ = [zeros(nvox); 1e-40 * ones(nvox); zeros(nvox); 0.001; 0.0; l; l]
# x₀ = [zeros(nvox); 1e-40 * ones(nvox); 0.001; 0.0; l; l]
x₀ = [zeros(2nvox); l]
# x̄ = [(0.005 * l^2 * ρ) * cₚ * T∞ * ones(nvox); (0.005 * l^2 * ρ) * ones(nvox); 0.4 * ones(nvox); l*ncols; l*nrows; l; l]
# x̄ = [(0.005 * l^2 * ρ) * cₚ * T∞ * ones(nvox); (0.005 * l^2 * ρ) * ones(nvox); l*ncols; l*nrows; l; l]
x̄ = [(0.001 * l^2 * ρₘ * cₚₘ) * T∞ * ones(nvox); ones(nvox); l]
# ū = [0.0; 0.0; 1.0; 0.0059]
ū = [0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

# Nx, Ny, l should be moved to the transfer process
Nkb = 180
Nkc = 1220
Nc = 1
Δtb = 0.01
Δtc = 0.01
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=x̄, Δtb=Δtb, Δtc=Δtc, final_constraint=false)

z₀ = generate_wall_z₀(process, problem.idx, x₀, Δtb, Δtc; free_time=false)
X = vcat([[z₀[problem.idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
U = vcat([[z₀[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)


animate_state_history(X, Δtb, nrows, ncols, strid=4, path="animation_state_prmotx.mp4")

function temperature(X, N)
    s = @view X[1:2N]
    T = zeros(N)

    temperature!(process.transfer_dynamics, T, s)
    return T
end

Y = [temperature(X[k], nvox) for k in 1:length(X)]

animate_measurement_history(Y, Δtb, nrows, ncols, strid=4, path="animation_measured_prmotx.mp4")

# z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, c_tol=1.0e-6, xg=x₀, ug=[0.0677])

# animate_state_history(X, Δtb, nrows, ncols, strid=4, path="animation_state_prmot_optimized_nohess.mp4")
# Y = [temperature(X[k], nvox) for k in 1:length(X)]
# animate_measurement_history(Y, Δtb, nrows, ncols, strid=4, path="animation_measured_prmot_optimized_nohess.mp4")