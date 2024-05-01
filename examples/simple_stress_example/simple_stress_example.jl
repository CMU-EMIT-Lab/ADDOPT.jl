
using LinearAlgebra
using ADDOPT
using Plots

Tₘₐₓ = [3000.0; 3000.0] # K
max_power = [5000.0; 0.0] # W
min_power = [0.0; 0.0]
total_power = 5000.0
h = 300.0 # W / m²K
T∞ = 293.15 # K
L₀ = 0.5 # m
A1 = 0.010^2 # m²
A2 = 0.010^2 # m²
P1 = 0.010*4 # m
P2 = 0.010*4 # m
ρ = 7826.0 # kg / m³
cₚ = 502.416 # J / kg K
# σy = 220e6 # Pa
E = 200e-3 # MPa / μstrain
α = 10.8 # μstrain / K

process = TwoBar(Tₘₐₓ, max_power, min_power, total_power, h, T∞, L₀, A1, A2, P1, P2, ρ, cₚ, E, α)

Q = Diagonal([0.0; 0.0; 1.0e-4; 1.0e-4; 1.0e-4; 1.0e-4; 0.0; 0.0; 0.0; 0.0])
R = Diagonal([1e-7; 1e-7])
Qf = 10 * Q
x₀ = [T∞; T∞; Inf; Inf; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x̄ = [T∞; T∞; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
ū = [0.0; 0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nk = 250
Nc = 1
problem = AdditiveProblem(process, objective, Nk, 4Nk, Nc, x₀; ximin=[T∞; T∞; -Inf; -Inf; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0], Δtb=0.1, Δtc=0.1, hessian=false)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; xg=[T∞+720; T∞+200; -3800; 3200; 20.0; 10.0; 10.0; -10.0; 30.0; -20.0], max_iter=5000, c_tol=1.0e-6)

t = cumsum(Δt) .- Δt[1]
T1 = [x[1] for x in X]
T2 = [x[2] for x in X]
σ1 = [x[3] for x in X] .* process.transfer_dynamics.E
σ2 = [x[4] for x in X] .* process.transfer_dynamics.E
ε_P1 = [x[5] for x in X]
ε_P2 = [x[6] for x in X]
Power_1 = [u[1] for u in U]
Power_2 = [u[2] for u in U]

σ1y = [σy(process.transfer_dynamics, x[1]) for x in X] 
σ2y = [σy(process.transfer_dynamics, x[2]) for x in X] 

p = plot(title="Temperature - Stress Evolution", xlabel="Time (s)", 
ylabel="Temperature (K)", ylim=(275, 1000))
plot!(p, t, T1, label="T1 (K)", linewidth=2)
plot!(p, t, T2, label="T2 (K)", linewidth=2)
tp = twinx(p)
ylabel!(tp, "Stress (MPa)", ylim=(-220, 220))
plot!(tp, t, σ1, label="σ1 (MPa)",linestyle=:dash, linewidth=2)
plot!(tp, t, σ2, label="σ2 (MPa)",linestyle=:dash, linewidth=2)
savefig("temp_stress_naive.svg")
savefig("temp_stress_naive.png")

plot(title="Stress Evolution", xlabel="Time (s)")
plot!(t, σ1, label="σ1 (MPa)")
plot!(t, σ2, label="σ2 (MPa)")
plot!(t, σ1y,  label="σ1y (MPa)")
plot!(t, σ2y,  label="σ2y (MPa)")
savefig("stress_naive.svg")
savefig("stress_naive.png")

plot(title="Plastic Strain", xlabel="Time (s)")
plot!(t, ε_P1, label="ε_P1 (μstrain)")
plot!(t, ε_P2, label="ε_P2 (μstrain)")
savefig("plastic_strain_naive.svg")
savefig("plastic_strain_naive.png")

p1 = plot( ylabel="Power (W)", ylims=(0, 5200), grid=false)
plot!(p1, t, Power_1, label=nothing, linewidth=1.5, linestyle=:dash)
plot!(p1, t, Power_2, label=nothing, linewidth=1.5, linestyle=:dash)
tp = twinx(p1)
ylims!(tp, (275, 800))
ylabel!(tp, "Temperature (K)")
plot!(tp, t, T1, label=nothing, linewidth=1.5)
plot!(tp, t, T2, label=nothing, linewidth=1.5)
# savefig(p1, "power_temp_naive.svg")
# savefig(p1, "power_temp_naive.png")

p2 = plot(xlabel="Time (s)", ylabel="Stress (MPa)", grid=false)
hline!([0.0,], label=nothing, color=:black, linewidth=0.5)
plot!(p2, t, σ1, ylims=(-200, 200), label=nothing, color=:purple, linewidth=1.5)
hline!([σ1[end],], label=nothing, linestyle=:dashdot, color=:black, linewidth=1.5)

pc = plot(p1, p2, layout=(2,1), size=(400, 400))
savefig(pc, "power_temp_stress_naive.svg")
savefig(pc, "power_temp_stress_naive.png")


plot(title="Temperature - Stress Evolution", xlabel="T (K)", ylabel="σ (MPa)", xlim=(275, 800), ylim=(-220, 220))
plot!(T1, σ1, label="Bar 1", linewidth=2)
plot!(T2, σ2, label="Bar 2", linewidth=2)
savefig("temp_stress_phase_naive.svg")
savefig("temp_stress_phase_naive.png")

σ1n = σ1[end]
σ2n = σ2[end]
Jn = J

Q = Diagonal([0.0; 0.0; 1.0e-4; 1.0e-4; 1.0e-4; 1.0e-4; 0.0; 0.0; 0.0; 0.0])
R = Diagonal([1e-7; 1e-7])
Qf = 1000 * Q
x₀ = [T∞; T∞; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x̄ = [T∞; T∞; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
ū = [0.0; 0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Tₘₐₓ = [3000.0; 3000.0] # K
max_power = [5000.0; 5000.0] # W
min_power = [0.0; 0.0]
process2 = TwoBar(Tₘₐₓ, max_power, min_power, total_power, h, T∞, L₀, A1, A2, P1, P2, ρ, cₚ, E, α)
problem2 = AdditiveProblem(process2, objective, Nk, 4Nk, Nc, x₀; Δtb=0.1, Δtc=0.1, hessian=false)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem2; xg=[T∞+700; T∞+270; -3400; 3200; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0], max_iter=5000, c_tol=1.0e-6)

t = cumsum(Δt) .- Δt[1]
T1 = [x[1] for x in X]
T2 = [x[2] for x in X]
σ1 = [x[3] for x in X] .* process.transfer_dynamics.E
σ2 = [x[4] for x in X] .* process.transfer_dynamics.E
ε_P1 = [x[5] for x in X]
ε_P2 = [x[6] for x in X]
Power_1 = [u[1] for u in U]
Power_2 = [u[2] for u in U]

σ1y = [σy(process.transfer_dynamics, x[1]) for x in X] 
σ2y = [σy(process.transfer_dynamics, x[2]) for x in X] 

p = plot(title="Temperature - Stress Evolution", xlabel="Time (s)", 
ylabel="Temperature (K)", ylim=(275, 1000))
plot!(p, t, T1, label="T1 (K)", linewidth=2)
plot!(p, t, T2, label="T2 (K)", linewidth=2)
tp = twinx(p)
ylabel!(tp, "Stress (MPa)", ylim=(-220, 220))
plot!(tp, t, σ1, label="σ1 (MPa)",linestyle=:dash, linewidth=2)
plot!(tp, t, σ2, label="σ2 (MPa)",linestyle=:dash, linewidth=2)
savefig("temp_stress_unconoptimized.svg")
savefig("temp_stress_unconoptimized.png")

plot(title="Stress Evolution", xlabel="Time (s)")
plot!(t, σ1, label="σ1 (MPa)")
plot!(t, σ2, label="σ2 (MPa)")
plot!(t, σ1y,  label="σ1y (MPa)")
plot!(t, σ2y,  label="σ2y (MPa)")
savefig("stress_unconoptimized.svg")
savefig("stress_unconoptimized.png")

plot(title="Plastic Strain", xlabel="Time (s)")
plot!(t, ε_P1, label="ε_P1 (μstrain)")
plot!(t, ε_P2, label="ε_P2 (μstrain)")
savefig("plastic_strain_unconoptimized.svg")
savefig("plastic_strain_unconoptimized.png")

p1 = plot( ylabel="Power (W)", ylims=(0, 5200), grid=false)
plot!(p1, t, Power_1, label=nothing, linewidth=1.5, linestyle=:dash)
plot!(p1, t, Power_2, label=nothing, linewidth=1.5, linestyle=:dash)
tp = twinx(p1)
ylims!(tp, (275, 800))
ylabel!(tp, "Temperature (K)")
plot!(tp, t, T1, label=nothing, linewidth=1.5)
plot!(tp, t, T2, label=nothing, linewidth=1.5)
# savefig(p1, "power_temp_unconoptimized.svg")
# savefig(p1, "power_temp_unconoptimized.png")

p2 = plot(xlabel="Time (s)", ylabel="Stress (MPa)", grid=false)
hline!([0.0,], label=nothing, color=:black, linewidth=0.5)
plot!(p2, t, σ1, ylims=(-200, 200), label=nothing, color=:purple, linewidth=1.5)
hline!([σ1[end],], label=nothing, linestyle=:dashdot, color=:black, linewidth=1.5)

pc = plot(p1, p2, layout=(2,1), size=(400, 400))
savefig(pc, "power_temp_stress_unconoptimized.svg")
savefig(pc, "power_temp_stress_unconoptimized.png")

plot(title="Temperature - Stress Evolution", xlabel="T (K)", ylabel="σ (MPa)", xlim=(275, 800), ylim=(-220, 220))
plot!(T1, σ1, label="Bar 1", linewidth=2)
plot!(T2, σ2, label="Bar 2", linewidth=2)
savefig("temp_stress_phase_unconoptimized.svg")
savefig("temp_stress_phase_unconoptimized.png")

σ1u = σ1[end]
σ2u = σ2[end]
Ju = J

Q = Diagonal([0.0; 0.0; 1.0e-4; 1.0e-4; 1.0e-4; 1.0e-4; 0.0; 0.0; 0.0; 0.0])
R = Diagonal([1e-7; 1e-7])
Qf = 1000 * Q
x₀ = [T∞; T∞; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x̄ = [T∞; T∞; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
ū = [0.0; 0.0]
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Tₘₐₓ = [3000.0; 450.0] # K
max_power = [5000.0; 5000.0] # W
min_power = [0.0; 0.0]
process2 = TwoBar(Tₘₐₓ, max_power, min_power, total_power, h, T∞, L₀, A1, A2, P1, P2, ρ, cₚ, E, α)
problem2 = AdditiveProblem(process2, objective, Nk, 4Nk, Nc, x₀; Δtb=0.1, Δtc=0.1, hessian=false)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem2; xg=[T∞+700; T∞+270; -3400; 3200; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0], max_iter=5000, c_tol=1.0e-6)

t = cumsum(Δt) .- Δt[1]
T1 = [x[1] for x in X]
T2 = [x[2] for x in X]
σ1 = [x[3] for x in X] .* process.transfer_dynamics.E
σ2 = [x[4] for x in X] .* process.transfer_dynamics.E
ε_P1 = [x[5] for x in X]
ε_P2 = [x[6] for x in X]
Power_1 = [u[1] for u in U]
Power_2 = [u[2] for u in U]

σ1y = [σy(process.transfer_dynamics, x[1]) for x in X] 
σ2y = [σy(process.transfer_dynamics, x[2]) for x in X] 

p = plot(title="Temperature - Stress Evolution", xlabel="Time (s)", 
ylabel="Temperature (K)", ylim=(275, 1000))
plot!(p, t, T1, label="T1 (K)", linewidth=2)
plot!(p, t, T2, label="T2 (K)", linewidth=2)
tp = twinx(p)
ylabel!(tp, "Stress (MPa)", ylim=(-220, 220))
plot!(tp, t, σ1, label="σ1 (MPa)",linestyle=:dash, linewidth=2)
plot!(tp, t, σ2, label="σ2 (MPa)",linestyle=:dash, linewidth=2)
savefig("temp_stress_optimized.svg")
savefig("temp_stress_optimized.png")

plot(title="Stress Evolution", xlabel="Time (s)")
plot!(t, σ1, label="σ1 (MPa)")
plot!(t, σ2, label="σ2 (MPa)")
plot!(t, σ1y,  label="σ1y (MPa)")
plot!(t, σ2y,  label="σ2y (MPa)")
savefig("stress_optimized.svg")
savefig("stress_optimized.png")

plot(title="Plastic Strain", xlabel="Time (s)")
plot!(t, ε_P1, label="ε_P1 (μstrain)")
plot!(t, ε_P2, label="ε_P2 (μstrain)")
savefig("plastic_strain_optimized.svg")
savefig("plastic_strain_optimized.png")


p1 = plot( ylabel="Power (W)", ylims=(0, 5200), grid=false)
plot!(p1, t, Power_1, label=nothing, linewidth=1.5, linestyle=:dash)
plot!(p1, t, Power_2, label=nothing, linewidth=1.5, linestyle=:dash)
tp = twinx(p1)
ylims!(tp, (275, 800))
ylabel!(tp, "Temperature (K)")
plot!(tp, t, T1, label=nothing, linewidth=1.5)
plot!(tp, t, T2, label=nothing, linewidth=1.5)
# savefig(p1, "power_temp_optimized.svg")
# savefig(p1, "power_temp_optimized.png")

p2 = plot(xlabel="Time (s)", ylabel="Stress (MPa)", grid=false)
hline!([0.0,], label=nothing, color=:black, linewidth=0.5)
plot!(p2, t, σ1, ylims=(-200, 200), label=nothing, color=:purple, linewidth=1.5)
hline!([σ1[end],], label=nothing, linestyle=:dashdot, color=:black, linewidth=1.5)

pc = plot(p1, p2, layout=(2,1), size=(400, 400))
savefig(pc, "power_temp_stress_optimized.svg")
savefig(pc, "power_temp_stress_optimized.png")

plot(title="Temperature - Stress Evolution", xlabel="T (K)", ylabel="σ (MPa)", xlim=(275, 800), ylim=(-220, 220))
plot!(T1, σ1, label="Bar 1", linewidth=2)
plot!(T2, σ2, label="Bar 2", linewidth=2)
savefig("temp_stress_phase_optimized.svg")
savefig("temp_stress_phase_optimized.png")

σ1o = σ1[end]
σ2o = σ2[end]

Jo = J
∂J∂Tmax2 = sum(vcat([[p[2] for p in u] for u in μ_xᵤ]...))

Tₘₐₓ = [3000.0; 455.0] # K
max_power = [5000.0; 5000.0] # W
min_power = [0.0; 0.0]
process2 = TwoBar(Tₘₐₓ, max_power, min_power, total_power, h, T∞, L₀, A1, A2, P1, P2, ρ, cₚ, E, α)
problem2 = AdditiveProblem(process2, objective, Nk, 4Nk, Nc, x₀; Δtb=0.1, Δtc=0.1, hessian=false)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem2; z₀=z.+randn(problem.idx.Nz), λ₀=λ, xg=[T∞+700; T∞+270; -3400; 3200; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0], max_iter=5000, c_tol=1.0e-6)

t = cumsum(Δt) .- Δt[1]
T1 = [x[1] for x in X]
T2 = [x[2] for x in X]
σ1 = [x[3] for x in X] .* process.transfer_dynamics.E
σ2 = [x[4] for x in X] .* process.transfer_dynamics.E
ε_P1 = [x[5] for x in X]
ε_P2 = [x[6] for x in X]
Power_1 = [u[1] for u in U]
Power_2 = [u[2] for u in U]

σ1y = [σy(process.transfer_dynamics, x[1]) for x in X] 
σ2y = [σy(process.transfer_dynamics, x[2]) for x in X] 

p = plot(title="Temperature - Stress Evolution", xlabel="Time (s)", 
ylabel="Temperature (K)", ylim=(275, 1000))
plot!(p, t, T1, label="T1 (K)", linewidth=2)
plot!(p, t, T2, label="T2 (K)", linewidth=2)
tp = twinx(p)
ylabel!(tp, "Stress (MPa)", ylim=(-220, 220))
plot!(tp, t, σ1, label="σ1 (MPa)",linestyle=:dash, linewidth=2)
plot!(tp, t, σ2, label="σ2 (MPa)",linestyle=:dash, linewidth=2)
savefig("temp_stress_optimized5.svg")
savefig("temp_stress_optimized5.png")

plot(title="Stress Evolution", xlabel="Time (s)")
plot!(t, σ1, label="σ1 (MPa)")
plot!(t, σ2, label="σ2 (MPa)")
plot!(t, σ1y,  label="σ1y (MPa)")
plot!(t, σ2y,  label="σ2y (MPa)")
savefig("stress_optimized5.svg")
savefig("stress_optimized5.png")

plot(title="Plastic Strain", xlabel="Time (s)")
plot!(t, ε_P1, label="ε_P1 (μstrain)")
plot!(t, ε_P2, label="ε_P2 (μstrain)")
savefig("plastic_strain_optimized5.svg")
savefig("plastic_strain_optimized5.png")

p1 = plot( ylabel="Power (W)", ylims=(0, 5200), grid=false)
plot!(p1, t, Power_1, label=nothing, linewidth=1.5, linestyle=:dash)
plot!(p1, t, Power_2, label=nothing, linewidth=1.5, linestyle=:dash)
tp = twinx(p1)
ylims!(tp, (275, 800))
ylabel!(tp, "Temperature (K)")
plot!(tp, t, T1, label=nothing, linewidth=1.5)
plot!(tp, t, T2, label=nothing, linewidth=1.5)
# savefig(p1, "power_temp_optimized5.svg")
# savefig(p1, "power_temp_optimized5.png")

p2 = plot(xlabel="Time (s)", ylabel="Stress (MPa)", grid=false)
hline!([0.0,], label=nothing, color=:black, linewidth=0.5)
plot!(p2, t, σ1, ylims=(-200, 200), label=nothing, color=:purple, linewidth=1.5)
hline!([σ1[end],], label=nothing, linestyle=:dashdot, color=:black, linewidth=1.5)

pc = plot(p1, p2, layout=(2,1), size=(400, 400))
savefig(pc, "power_temp_stress_optimized5.svg")
savefig(pc, "power_temp_stress_optimized5.png")

plot(title="Temperature - Stress Evolution", xlabel="T (K)", ylabel="σ (MPa)", xlim=(275, 800), ylim=(-220, 220))
plot!(T1, σ1, label="Bar 1", linewidth=2)
plot!(T2, σ2, label="Bar 2", linewidth=2)
savefig("temp_stress_phase_optimized5.svg")
savefig("temp_stress_phase_optimized5.png")

σ1o5 = σ1[end]
σ2o5 = σ2[end]

Jo5 = J

println("Naive stresses:\nσ1: $(σ1n) MPa, σ2: $(σ2n) MPa")
println("Unconstrained optimized stresses:\nσ1: $(σ1u) MPa, σ2: $(σ2u) MPa")
println("Constrained (T2 ≤ 450K) optimized stresses:\nσ1: $(σ1o) MPa, σ2: $(σ2o) MPa")
println("Jo: $Jo, ∂J∂Tmax2: $∂J∂Tmax2 1/K, Jpred for 5deg: $(Jo+∂J∂Tmax2*5)")
println("Constrained (T2 ≤ 455K) optimized stresses:\nσ1: $(σ1o5) MPa, σ2: $(σ2o5) MPa")
println("Jo5: $Jo5")