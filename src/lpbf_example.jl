include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, QuadraticObjective, optimize_trajectory, generate_wall_z₀, animate_state_history, input_idle, animate_measurement_history, constraints!, PlanarLPBF, PlanarLPBFPrescribedMotion
using Plots
using JLD2

A = 1e4
τ = 9625

σ = 1.2e-3#5e-4 (0.707 may be theory)

h∞ = 2000 # W / m^2 K
h₀ = 7500 # W / m^2 K
hₐᵣ = 500
η = 0.8#0.95
k(T) = 34 # W / mK ############# TEMPORARY
ρ = 7826.0e6 # kg / m^3 # mg
cₚ = 502.416e-6 # J / kg K # mg 
T∞ = 295.0 # K
T₀ = 295.0 # K
Tₗ = 1784.0 # K, liquidus

nrows = 6
ncols = 6
nvox = nrows * ncols
l = 0.001 # m, aka 1mm

process = PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, 500.0, 20.0)
# process = PlanarLPBFPrescribedMotion(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, l, l * ncols, l, l * nrows, 0.0, 1.8)

# Q = Diagonal([1e-6 * ones(nvox); 1e6 * ones(nvox); 1e-4 * ones(2)]) #1e4 mass in kg
Q = Diagonal([1e-6 * ones(nvox); 1e6 * ones(nvox)]) #1e4 mass in kg
# R = Diagonal([1e-2; 1e-2; 1e-6; 0 * ones(nvox)])
R = Diagonal(1e-6 * ones(nvox))

# R = Diagonal([1e-3])
Qf = 10 * Q
# x₀ = [T∞ * ones(nvox); zeros(nvox); 0.0; 0.0]
x₀ = [T∞ * ones(nvox); zeros(nvox)]
# x̄ = [T∞ * ones(nvox); 0.9 * ones(nvox); l * ncols; l * nrows]
x̄ = [T∞ * ones(nvox); 0.6 * ones(nvox)]
# ū = [0.0; 0.0; 20.0; zeros(nvox)]
ū = zeros(nvox)

# ū = [20.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

# Nx, Ny, l should be moved to the transfer process
Nkb = 180
Nkc = 320
Nc = 1
Δtb = 0.01
Δtc = 0.01
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=x̄, Δtb=Δtb, Δtc=Δtc, final_constraint=false)

# z₀ = generate_wall_z₀(process, problem.idx, x₀, Δtb, Δtc; free_time=false)
# X = vcat([[z₀[problem.idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
# U = vcat([[z₀[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)
# Y = [X[k][1:nvox] for k in 1:length(X)]

# animate_state_history(X, Δtb, nrows, ncols, strid=4, path="animation_state_lpbf.mp4")
# animate_measurement_history(Y, Δtb, nrows, ncols, strid=4, path="animation_measured_lpbf.mp4")

# z0 = copy(z)
z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, c_tol=1.0e-6, ug=30.0 * (ones(nvox) + randn(nvox)), xg=[1390 * ones(nvox); 0.0 * ones(nvox)])#, z₀=z0) 1200, .4

Y = [X[k][1:nvox] for k in 1:length(X)]
animate_state_history(X, Δtb, nrows, ncols, strid=1, path="animation_state_lpbf_optimized_gausshess_hightempguess_linear6var.mp4")
animate_measurement_history(Y, Δtb, nrows, ncols, strid=1, path="animation_measured_lpbf_optimized_gausshess_hightempguess_linear6var.mp4")
animate_measurement_history(U.*5, Δtb, nrows, ncols, strid=1, path="animation_power_lpbf_optimized_gausshess_hightempguess_linear6var.mp4")

save_object("traj_lpbf_gausshess_hightempguess_linear6var.jld2", z) #linear 4: 6x6, 80W max. Linear 5: 500W max. Linear 6: 500W, with variance constraint

xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

xt = [sum(xₙ.*U[k])/sum(U[k]) for k in 1:length(X)]
zt = [sum(zₙ.*U[k])/sum(U[k]) for k in 1:length(X)]

P = [sum(U[k][:]) for k in 1:length(U)]
plot(P)

# vx = [U[k][1] for k in 1:length(U)]
# vz = [U[k][2] for k in 1:length(U)]
# P = [U[k][3] for k in 1:length(U)]
plot(xt, zt)