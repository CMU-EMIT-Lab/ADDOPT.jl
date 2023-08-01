include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, CoolingTrackingObjective, TimeWeightedQuadraticObjective, optimize_trajectory, generate_wall_z₀, animate_state_history, input_idle, animate_measurement_history, state_min, property_min, temperature!, WAAMPrescribedMotion, WAAMHardnessPrescribedMotion, VoxelEnergyFillDynamics, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar, gen_knots, gen_temp_ref, gen_objective_weights, gen_fill_ref, gen_xyz, rollout, marshall_z, QuadraticTrackingObjective, gen_ref, linear2exp, gen_exponential, ThermalICProblem, optimize_thermal_ic
using Plots
using JLD2

A = 1e4
τ = 9625
# σ = 5.670374419 * 10^(-8) # Stefan-Boltzmann, W / (m⁴⋅ K⁴)

h∞ = 10 # W / m^2 K
h₀ = 7500 # W / m^2 K
hₐᵣ = 500
η = 0.8#0.95

k = 34.0 # W / mK
ρ = 7826.0e6 # kg / m^3 # mg
cₚ = 502.416e-6 # J / kg K # mg 

T∞ = 295.0 # K
T₀ = 295.0 # K
wire_diam = 0.001143 # m, aka 0.045in
Tₗ = 1784.0 # K, liquidus

nx = 10
ny = 5
nz = 3
nvox = nz * ny * nx

Nkb = 100
Nkc = 500
Nc = 1
Δtb = 0.02
Δtc = 0.02

Tmin = T₀
Tmax = 3000.0 # K

l = 0.001 # m, aka 1mm

xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)

layers = [[[l; (ny + 1) / 2 * l; 1l], [nx * l; (ny + 1) / 2 * l; 1l]], [[l; (ny + 1) / 2 * l; 1l], [nx * l; (ny + 1) / 2 * l; 1l]]]
p̄ = gen_knots(layers, Nkb, Nkc, Nc)
fill_ref = gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=2.0e-3)

x̄ = [clamp.(Float64.(f), 0.0, 1.0) for f in fill_ref]

for k in 1:lastindex(fill_ref)
    clamp!(fill_ref[k], 0, Nkb) # 0, Nkb
    fill_ref[k][fill_ref[k].>0] .= Nkb
end

process = WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄)

function temperature(X, N)
    s = @view X[1:2N]
    T = zeros(N)

    temperature!(process.transfer_dynamics, T, s)
    return T
end

x₀ = zeros(2nvox)
ȳ = 0.5 * x̄[end]
Qic = 10.0 * Diagonal(ones(nvox))
Eₘᵢₙ = 1000.0 * (l^3 * ρ * cₚ) * x̄[end]
Eₘₐₓ = Tₗ * (l^3 * ρ * cₚ) * x̄[end]
J = zeros(nvox, nvox)
icprob = ThermalICProblem(process2, nvox, x̄[end], zeros(nvox), Eₘᵢₙ, Eₘₐₓ, Δtc, Nkc, ȳ, Qic, J)
Ei = optimize_thermal_ic(icprob; tol=1e-4, c_tol=1e-4)

Q = Diagonal([ones(nvox); 20 * ones(nvox)])
Qf = Nkb * Δtb * Q
xg = [Ei; x̄[end]]
objective = TimeWeightedQuadraticObjective(Q, Qf, xg)
problem = AdditiveProblem(process, objective, Nkb, 50, Nc, x₀, x̄=xg, Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=false);

X0 = [[[(l^3 * ρ * cₚ) * T∞ * clamp.(x .- 0.4, 0.0, 1.0); clamp.(x .- 0.4, 0.0, 1.0)] for x in x̄]]
U0 = [0.030 * ones(Nkb)]
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc; free_time=true)

z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, c_tol=1.0e-5, z₀=z0)
Y = [temperature(x, nvox) for x in X]

t = cumsum(Δt)
save_object("traj_waam3d10_hardness_prescribed.jld2", z)

slow_fac = 20
Nkb_sim = slow_fac * Nkb
Nkc_sim = slow_fac * Nkc
Δtb_sim = Δtb / (Nkb_sim / Nkb)
Δtc_sim = Δtc / (Nkc_sim / Nkc)

# NOT RESAMPLING CORRECTLY RN (not taking acount of Δt changes)
p̄_sim = gen_knots(layers, Nkb_sim, Nkc_sim, Nc)
process2 = WAAMHardnessPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄_sim, A, τ)

using Interpolations
uinterp = linear_interpolation(t[1:Nkb], [u[1] for u in U], extrapolation_bc=Flat())
t_sim = range(0, (Nkb_sim-1)*Δtb_sim, length=Nkb_sim)
U_sim = uinterp.(t_sim)

X_sim = rollout(process2, zeros(3nvox), [U_sim,], Nkb_sim, Nkc_sim, Nc, Δtb_sim, Δtc_sim, free_time=false)
X_sim = vcat(X_sim...)
animate_3Dstate_history_planar(X_sim, Δtb_sim, nx, ny, nz; path="animation_state3d10_hardness_optimized.mp4", strid=slow_fac)
Y = [temperature(x, nvox) for x in X_sim]
animate_3Dmeasurement_history_planar(Y, X_sim, Δtb_sim, nx, ny, nz; path="animation_measured3d10_hardness_optimized.mp4", strid=slow_fac)
Y = [x[(2nvox+1):3nvox] for x in X_sim]
animate_3Dmeasurement_history_planar(Y, X_sim, Δtb_sim, nx, ny, nz; path="animation_measured_frac_3d10_hardness_optimized.mp4", strid=slow_fac, quantity="Fraction Transformed", scale=(0, 1))

plot(t, [u[1] for u in U], xlabel="Time (s)", ylabel="Wire Feed Speed (m/s)", label="ȳ=0.5")
# 1: 0.1, 2: 0.4 fixed time
# 3: 0.1, 4: 0.4 free build time
# 5: direct exponential, free time, weights borked
# 6: direct exponential, free time, weights fixed, time bounds borked, weights still probably need adjustment
# 7: adjusted time bounds
# 8: adjusted input bounds
# 9: Direct time-weighted quadratic on goal, ȳ=0.1
# 10: Direct time-weighted quadratic on goal, ȳ=0.5


# plot(t, [y[46] for y in Y], xlabel="Time (s)", ylabel="Temperature (K)", label="Voxel 46")
# plot!(t, clamp.(T̄.(t .- t[Nkb]), 0, Tmax), label="Reference Shifted")

# plot(t, [u[1] for u in U] .* (π / 4 * wire_diam^2) ./ (0.0001 ./ Δt))
# plot(t, 0.0001 ./ Δt, xlabel="Time (s)", ylabel="Travel Speed (m/s)")
WFS = [u[1] for u in U]
TS = 0.0001 ./ Δt
r = @. wire_diam * √(WFS / (2TS))
plot(t, r, xlabel="Times (s)", ylabel="Bead Radius (m)")