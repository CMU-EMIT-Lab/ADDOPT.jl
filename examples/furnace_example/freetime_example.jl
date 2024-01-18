include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: FurnaceSimple, AdditiveProblem, QuadraticObjective, optimize_trajectory, marshall_z
@time using Plots

Pₘₐₓ = 2000.0  # W
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 1.0      # W/m²K
m = 0.1        # kg
cₚ = 502.416   # J/kg 
process = FurnaceSimple(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

Q = Diagonal([1e0])
R = Diagonal([0e-3])
Qf = 10 * Q
x₀ = [T∞]
x̄ = [700.0] #[350.0]
ū = [0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

# Nx, Ny, l should be moved to the transfer process
Nk = 250
Nc = 3
problem = AdditiveProblem(process, objective, Nk, 3Nk, Nc, x₀, x̄=x̄, Δtb=nothing, Δtc=0.04, final_constraint=false)

z, X, U, Δt, λ = optimize_trajectory(problem; max_iter=2000, c_tol=1.0e-6, xg=[300.0])
Nt = length(X)
T = [X[i][1] for i in 1:Nt]
# y = [X[i][2] for i in 1:Nt]
P = [U[i][1] for i in 1:Nt]

if isnothing(problem.Δtb)
    t = cumsum(Δt)
else
    t = (1:Nt) .* problem.Δtb
end

plot(t, T, label="Temperature (K)", xlabel="Time (s)", color="orange")
plot!(t, P ./ 10, label="Power (W)", color="purple")
# savefig("freetime3.png")
# plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing)