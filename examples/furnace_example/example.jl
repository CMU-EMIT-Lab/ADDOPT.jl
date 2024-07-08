using LinearAlgebra
using ADDOPT
using Plots

Pₘₐₓ = 800e-3  # kW
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 1.0e-3      # kW/m² K
m = 0.01        # kg
cₚ = 502.416e-3   # kJ/kg K 
lnA = 38.005
n = 0.051590
E = 240.24 # kJ / mol K
process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ, lnA, n, E)

Q = Diagonal([0.0; 1e0])
R = Diagonal([1e-4])
b = zeros(2)
Qf = 10 * Q
x₀ = [T∞; log(-log(1-0.1))]
x̄ = [T∞; log(-log(1-0.9))]
ū = [0.0]
objective = QuadraticObjective(Q, b, R, Qf, x̄, ū)

Nk = 250
Nc = 3
problem = AdditiveProblem(process, objective, [Nk ÷ 2; Nk ÷ 2; Nk], [Nk; Nk; 3Nk], Nc, x₀, x̄=[300.0; Inf], Δtb=0.04, Δtc=0.04, final_constraint=true, xfmin=[0.0; -Inf], hessian=true)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=1000, c_tol=1.0e-6, xg=[600.0; log(-log(1-0.4))])
Nt = length(X)
T = [X[i][1] for i in 1:Nt]
y = [1 - exp(-exp(X[i][2])) for i in 1:Nt]
P = [U[i][1]*1e3 for i in 1:Nt]

t = cumsum(Δt)

plot(xlabel="Time (s)", ylabel="Temperature (K) & Power (W)", ylims=(0, 1200), grid=false, size=(450, 320))
plot!(t, T, label=nothing, color="orange") #"Temperature (K)"
plot!(t, P, label=nothing, color="purple") #"Power (W)"
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing, ylims=(0, 1)) #"Phase Fraction"