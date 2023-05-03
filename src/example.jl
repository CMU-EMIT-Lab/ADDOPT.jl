include("ADOPT.jl")
using LinearAlgebra
using .ADOPT: Furnace, AdditiveProblem, QuadraticObjective, optimize_trajectory
using Plots

Pₘₐₓ = 20000.0  # W
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 50.0      # W/m²K
m = 0.1        # kg
cₚ = 502.416   # J/kg 
process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

Q = diagm([1e-7; 1e6])
R = 1e-5*I
Qf = 10*Q
x₀ = [T∞; 0.0]
x̄ = [T∞; 0.8]
ū = [0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū, 1e-6)

# Nx, Ny, l should be moved to the transfer process
Nk = 2000
problem = AdditiveProblem(process, objective, Nk, 1, x₀; x̄=x̄)

z, X, U, Δt, tc = optimize_trajectory(problem; max_iter=1000)
T = [X[1][i][1] for i in 1:Nk]
y = [X[1][i][2] for i in 1:Nk]
P = [U[1][i][1] for i in 1:Nk]

t = (1:Nk) .* Δt[1]
# plot(t, P, label="Power (W)", xlabel="Time (s)")
plot(t, T, label="Temperature (K)", xlabel="Time (s)", color="orange")
# plot!(t, P, label="Power (W)", color="purple")
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing)#, xlabel="Time (s)")