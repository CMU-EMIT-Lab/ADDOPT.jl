include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, QuadraticObjective, TimeWeightedQuadraticObjective, optimize_trajectory, generate_wall_z₀, temperature!, WAAMPrescribedMotion, WAAMHardnessPrescribedMotion, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar, gen_knots, gen_fill_ref, gen_xyz, rollout, marshall_z, ThermalICProblem, optimize_thermal_ic, resample_vector_traj, WAAMHardnessCooling
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

nx = 10#50
ny = 5
nz = 3
nvox = nz * ny * nx

Nkb = 100#500
Nkc = 25
Nc = 1
Δtb = 0.02
Δtc = 0.02

Tmin = T₀
Tmax = 3000.0 # K

l = 0.001 # m, aka 1mm

xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)

layers = [[[3l; (ny + 1) / 2 * l; 1l], [(nx - 2) * l; (ny + 1) / 2 * l; 1l]], [[3l; (ny + 1) / 2 * l; 1l], [(nx - 2) * l; (ny + 1) / 2 * l; 1l]]]
p̄ = gen_knots(layers, Nkb, Nkc, Nc)
fill_ref = gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=2.1e-3)

x̄ = [clamp.(Float64.(f), 0.0, 1.0) for f in fill_ref]

process = WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄)

function temperature(X, N)
    s = @view X[1:2N]
    T = zeros(N)

    temperature!(process.transfer_dynamics, T, s)
    return T
end

x₀ = zeros(2nvox)
ȳ = 0.1
Qic = 10.0 * Diagonal(ones(nvox))
Eₘᵢₙ = 1000.0 * (l^3 * ρ * cₚ) * x̄[end]
Eₘₐₓ = Tₗ * (l^3 * ρ * cₚ) * x̄[end]
J = zeros(nvox, nvox)
process2 = WAAMHardnessCooling(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ)
icprob = ThermalICProblem(process2, nvox, x̄[end], zeros(nvox), Eₘᵢₙ, Eₘₐₓ, Δtc, 500, ȳ * x̄[end], Qic, J)
Ei = optimize_thermal_ic(icprob; tol=1e-5, c_tol=1e-5)

# ȳ = 0.1
# Qic = Diagonal([1e-4 * ones(nvox); 1e-2 * ones(nvox); 1e2 * ones(nvox)])
# Qicf = 10 * Qic
# x̄ic = [T∞ * (l^3 * ρ * cₚ) * x̄[end]; x̄[end]; ȳ * x̄[end]]

# Δtic = 0.02
# x0icmin = [1000.0 * (l^3 * ρ * cₚ) * x̄[end]; x̄[end]; zeros(nvox)]
# x0icmax = [1750.0 * (l^3 * ρ * cₚ) * x̄[end]; x̄[end]; zeros(nvox)]
# x0icmid = (x0icmin + x0icmax) / 2
# x0icguess = [900.0 * (l^3 * ρ * cₚ) * x̄[end]; x̄[end]; zeros(nvox)]

# process3 = WAAMHardnessCooling(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ)
# icobj = QuadraticObjective(Qic, Diagonal([1.0]), Qicf, x̄ic, [0.0])
# icprob = AdditiveProblem(process3, icobj, 1, 399, 1, x0icmax; x̄=x̄ic, Δtb=Δtic, Δtc=Δtic, ximin=x0icmin)

# # X0ic = [[x0icmid * (1 - α) + α * x̄ic for α in range(0, 1, length=400)],]
# X0ic = [[x0icguess for k in 1:400],]
# U0ic = [[0.0],]
# z0ic = marshall_z(icprob.idx, X0ic, U0ic, Δtic, Δtic; free_time=false)
# zic, Xic, Uic, Δtic = optimize_trajectory(icprob; z₀=z0ic)#; solv="ma77")
# Ei = Xic[1][(nvox+1):2nvox]

Q = Diagonal([1e-2 * ones(nvox); 20e-2 * ones(nvox)])
xg = [Ei; x̄[end]]
objective = TimeWeightedQuadraticObjective(Q, xg)
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=xg, Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=false,
    Δtb_min=0.01, Δtb_max=0.04,
    Δtc_min=0.01, Δtc_max=0.04)

X0 = [[[(l^3 * ρ * cₚ) * T∞ * clamp.(x .- 0.4, 0.0, 1.0); clamp.(x .- 0.4, 0.0, 1.0)] for x in x̄]]
U0 = [0.030 * ones(Nkb)]
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc; free_time=true)

z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, tol=1e-5, c_tol=1.0e-5, z₀=z0, solv="ma97")
Y = [temperature(x, nvox) for x in X]

t = cumsum(Δt)
save_object("traj_waam3d13_hardness_prescribed.jld2", z)

slow_fac = 20
Δt_sim = Δtb / slow_fac
t_sim = collect(range(0, t[end] + 20.0, step=Δt_sim))
Nk_sim = length(t_sim)

p̄_sim = resample_vector_traj(t, p̄, t_sim)
process2 = WAAMHardnessPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄_sim, A, τ)

U_sim = resample_vector_traj(t, U, t_sim)
X_sim = resample_vector_traj(t, X, t_sim)
# Append for final cooling, to time and input

X_sim = rollout(process2, zeros(3nvox), U_sim, Nk_sim, Δt_sim)
animate_3Dstate_history_planar(X_sim, Δt_sim, nx, ny, nz; path="animation_state3d13_hardness_optimized.mp4", strid=slow_fac)
Y = [temperature(x, nvox) for x in X_sim]
animate_3Dmeasurement_history_planar(Y, X_sim, Δt_sim, nx, ny, nz; path="animation_measured3d13_hardness_optimized.mp4", strid=slow_fac)
Y = [x[(2nvox+1):3nvox] for x in X_sim]
animate_3Dmeasurement_history_planar(Y, X_sim, Δt_sim, nx, ny, nz; path="animation_measured_frac_3d13_hardness_optimized.mp4", strid=slow_fac, quantity="Fraction Transformed", scale=(0, 1))

# 11 0.4 bad objectives
# 12 0.4
# 13 0.1

WFS = [u[1] for u in U]
plot(t, WFS, xlabel="Time (s)", ylabel="Wire Feed Speed (m/s)", label="ȳ=$(ȳ)")

TS = 0.0001 ./ Δt
r = @. wire_diam * √(WFS / (2TS))
plot(t, r, xlabel="Times (s)", ylabel="Bead Radius (m)")