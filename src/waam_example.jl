include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, QuadraticObjective, optimize_trajectory, generate_wall_z₀, animate_state_history, input_idle, animate_measurement_history, state_min, property_min, temperature!, WAAMPrescribedMotion, VoxelEnergyFillDynamics, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar
using Plots
using JLD2

# A = 1e4
# τ = 9625
# σ = 5.670374419 * 10^(-8) # Stefan-Boltzmann, W / (m⁴⋅ K⁴)

h∞ = 10 # W / m^2 K
h₀ = 7500 # W / m^2 K
hₐᵣ = 500
η = 0.95

k = 34.0 # W / mK
ρ = 7826.0e6 # kg / m^3 # mg
cₚ = 502.416e-6 # J / kg K # mg 

T∞ = 295.0 # K
T₀ = 295.0 # K
wire_diam = 0.001143 # m, aka 0.045in
Tₗ = 1784.0 # K, liquidus

nx = 10
ny = 6
nz = 3
nvox = nz * ny * nx

Tmin = T₀
Tmax = 1900 # K

l = 0.001 # m, aka 1mm

p̄ = [[l; (ny+1)/2*l; l], [nx*l; (ny+1)/2*l; l]]
t̄ = [0.0; 1.8]

process = WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, t̄)

xdist = zeros(nx, ny, nz)
xdist[:, 2:5, 1:3] .= 1
xdist = reshape(xdist, (nvox))

Q = Diagonal([1e-5 * ones(nvox); 1e0 * ones(nvox)]) #1e4 mass in kg
R = Diagonal(1e-2 * ones(1))
Qf = 10 * Q
x₀ = zeros(2nvox)
x̄ = [(l^3 * ρ * cₚ) * T∞ * xdist; xdist]
ū = [0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nkb = 90#180
Nkc = 360#720#1220
Nc = 1
Δtb = 0.02#0.01
Δtc = 0.02#0.01
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=x̄, Δtb=Δtb, Δtc=Δtc, final_constraint=false)

z₀ = generate_wall_z₀(process, problem.idx, x₀, Δtb, Δtc; free_time=false)
X = vcat([[z₀[problem.idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
U = vcat([[z₀[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)
 
# animate_3Dstate_history_planar(X, Δtb, nx, ny, nz; path="animation_state3d3.mp4", strid=2)

function temperature(X, N)
    s = @view X[1:2N]
    T = zeros(N)
 
    temperature!(process.transfer_dynamics, T, s)
    return T
end

Y = [temperature(X[k], nvox) for k in 1:length(X)]
# animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured3d3.mp4", strid=2)

z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, c_tol=1.0e-5, xg=x₀, ug=[0.0677]) #c_tol=1.0e-6
Y = [temperature(X[k], nvox) for k in 1:length(X)]

animate_3Dstate_history_planar(X, Δtb, nx, ny, nz; path="animation_state3d3_optimized.mp4", strid=2)
animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured3d3_optimized.mp4", strid=2)

save_object("traj_waam3d3_prescribed.jld2", z)
# for no. 3, loosened constraint tolerance to e-5 from e-6, doubled time step size, ma97