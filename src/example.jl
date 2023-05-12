include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: Furnace, AdditiveProblem, QuadraticObjective, optimize_trajectory
using Plots

Pₘₐₓ = 20000.0  # W
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 10.0      # W/m²K
m = 0.1        # kg
cₚ = 502.416   # J/kg 
process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

Q = Diagonal([1e-7; 1e6])
R = Diagonal([1e-9])
Qf = 10*Q
x₀ = [T∞; 0.0]
x̄ = [T∞+5; 0.8]
ū = [0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū, 1e4)

# Nx, Ny, l should be moved to the transfer process
Nk = 1000
problem = AdditiveProblem(process, objective, Nk, 1, x₀, x̄=x̄, Δt=0.04)

z, X, U, Δt, tc = optimize_trajectory(problem; max_iter=3000, c_tol=1.0e-6)
T = [X[1][i][1] for i in 1:Nk]
y = [X[1][i][2] for i in 1:Nk]
P = [U[1][i][1] for i in 1:Nk]

# t = (1:Nk) .* Δt[1]
# t = cumsum([Δt[1][i] for i in 1:Nk])
t = (1:Nk) .* problem.Δt
plot(t, T, label="Temperature (K)", xlabel="Time (s)", color="orange")
plot!(t, P./10, label="Power (W)", color="purple")
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing)#, xlabel="Time (s)")