include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: PlanarWAAMHardness, AdditiveProblem, QuadraticObjective, optimize_trajectory
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
# k = 34 # W / mK
ρ = 7826 # kg / m^3
cₚ = 502.416 # J / kg K
T∞ = 295.0 # K
T₀ = 295.0 # K
wire_diam = 0.001143 # m, aka 0.045in
Tₗ = 1784 # K, liquidus

nrows = 6
ncols = 20
# N = n_rows * n_cols
l = 0.001 # m, aka 1mm

process = PlanarWAAMHardness(nrows, ncols, l, xₙ, zₙ, k, ρ, cₚ, T∞, wire_diam, h∞, h₀, hₐᵣ, η, γᵣ, γₕ, wₓ, bₕ, A, τ)


Q = Diagonal([1e-7; 1e6])
R = Diagonal([1e-9])
Qf = 10 * Q
x₀ = [T∞; 0.0]
x̄ = [T∞ + 5; 0.8]
ū = [0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū, 1e4)

# Nx, Ny, l should be moved to the transfer process
Nk = 1000
problem = AdditiveProblem(process, objective, Nk, 1, x₀, x̄=x̄, Δt=0.05)

z, X, U, Δt, tc = optimize_trajectory(problem; max_iter=3000, c_tol=1.0e-6)
T = [X[1][i][1] for i in 1:Nk]
y = [X[1][i][2] for i in 1:Nk]
P = [U[1][i][1] for i in 1:Nk]

if isnothing(problem.Δt)
    t = cumsum([Δt[1][i] for i in 1:Nk])
else
    t = (1:Nk) .* problem.Δt
end

plot(t, T, label="Temperature (K)", xlabel="Time (s)", color="orange")
plot!(t, P ./ 10, label="Power (W)", color="purple")
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing)#, xlabel="Time (s)")