using LinearAlgebra
using ADDOPT
using Plots

Pₘₐₓ = 600.0  # W
T∞ = 293.15    # K
Tₘₐₓ = 900.0  # K
h = 1.0      # W/m²K
m = 0.01        # kg
cₚ = 502.416   # J/kg 
process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

Q = Diagonal([0.0; 1e0])
R = Diagonal([1e-8])
Qf = 10 * Q
x₀ = [T∞; 0.0]
x̄ = [T∞; 0.4]
ū = [0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nk = 250
Nc = 3
problem = AdditiveProblem(process, objective, [Nk÷2; Nk÷2; Nk], [Nk; Nk; 3Nk], Nc, x₀, x̄=[300.0; 1.0], Δtb=nothing, Δtc=nothing, final_constraint=true, xfmin=[0.0; 0.0], hessian=true)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=1000, c_tol=1.0e-6, xg=[600.0; 0.2])
Nt = length(X)
T = [X[i][1] for i in 1:Nt]
y = [X[i][2] for i in 1:Nt]
P = [U[i][1] for i in 1:Nt]

t = cumsum(Δt)

plot(xlabel="Time (s)", ylabel="Temperature (K) & Power (W)", ylims=(0, 1200), grid=false, size=(450, 320))
plot!(t, T, label=nothing, color="orange") #"Temperature (K)"
plot!(t, P, label=nothing, color="purple") #"Power (W)"
plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing, ylims=(0, 1)) #"Phase Fraction"
# savefig("furnace_default.svg")
# savefig("furnace_default.png")

# J0 = J
# ∂J∂Pₘₐₓ = sum(vcat([[p[1] for p in u] for u in μ_uᵤ]...))

# Pₘₐₓ = 630.0  # W
# T∞ = 293.15    # K
# Tₘₐₓ = 900.0  # K
# h = 1.0      # W/m²K
# m = 0.01        # kg
# cₚ = 502.416   # J/kg 
# process = Furnace(Pₘₐₓ, T∞, Tₘₐₓ, h, m, cₚ)

# Nk = 250
# Nc = 3
# problem = AdditiveProblem(process, objective, [Nk; Nk; Nk], [Nk; Nk; 3Nk], Nc, x₀, x̄=[700.0; 1.0], Δtb=0.04, Δtc=0.04, final_constraint=true, xfmin=[0.0; 0.0])

# J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=1000, c_tol=1.0e-6, xg=[600.0; 0.4])
# Nt = length(X)
# T = [X[i][1] for i in 1:Nt]
# y = [X[i][2] for i in 1:Nt]
# P = [U[i][1] for i in 1:Nt]

# t = cumsum(Δt)

# plot(xlabel="Time (s)", ylabel="Temperature (K) & Power (W)", ylims=(0, 1200), grid=false, size=(450, 320))
# plot!(t, T, label=nothing, color="orange") #"Temperature (K)"
# plot!(t, P, label=nothing, color="purple") #"Power (W)"
# plot!(twinx(), t, y, ylabel="Phase Fraction", label=nothing, ylims=(0, 1)) #"Phase Fraction"
# savefig("furnace_addact.svg")
# savefig("furnace_addact.png")

# @show J0
# @show ∂J∂Pₘₐₓ
# @show J0 + ∂J∂Pₘₐₓ * 30.0
# @show J

# plot(ylabel="Phase Fraction MSE", size=(450, 320), ytickfontsize=11, yguidefontsize=11)
# bar!(["Pₘₐₓ = 600W", "Pₘₐₓ = 630W"], [J0, J] ./ 1e8, color=[:red, :blue], label=nothing, xtickfontsize=11)
# plot!(["Pₘₐₓ = 600W", "Pₘₐₓ = 630W"], [J0, J0 + ∂J∂Pₘₐₓ * 30.0] ./ 1e8, linewidth=4, color=:purple, arrow=true, label="Prediction from Duals", legendfontsize=10)
# savefig("furnace_obj_change.svg")
# savefig("furnace_obj_change.png")