include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: Furnace, AdditiveProblem, QuadraticObjective, MinTimeObjective, optimize_trajectory, marshall_z
@time using Plots

Pₘₐₓ = 20000.0  # W
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 10.0      # W/m²K
m = 0.1        # kg
cₚ = 502.416   # J/kg 
process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

Q = Diagonal([1e-7; 1e6])
R = Diagonal([1e-7])
Qf = 10 * Q
x₀ = [T∞; 0.0]
x̄ = [T∞; 0.9]
ū = [0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

# Nx, Ny, l should be moved to the transfer process
Nk = 250
Nc = 3
problem = AdditiveProblem(process, objective, Nk, 3Nk, Nc, x₀, x̄=x̄, Δtb=0.04, Δtc=0.04, final_constraint=false)

# z0 = copy(z) + [repeat([10; 1],4Nk * Nc) .* randn(problem.idx.Nstates * 4Nk * Nc); 200 .* randn(Nk * Nc)]
z, X, U, Δt, λ = optimize_trajectory(problem; max_iter=1000, c_tol=1.0e-6)#, z₀=z0)#, λ₀=λ)
Nt = length(X)
T = [X[i][1] for i in 1:Nt]
y = [X[i][2] for i in 1:Nt]
P = [U[i][1] for i in 1:Nt]

if isnothing(problem.Δtb)
    t = cumsum(Δt)
else
    t = (1:Nt) .* problem.Δtb
end

plot(t, T, label="Temperature (K)", xlabel="Time (s)", color="orange")
plot!(t, P ./ 10, label="Power (W)", color="purple")
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing)