include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, CoolingTrackingObjective, optimize_trajectory, generate_wall_z₀, animate_state_history, input_idle, animate_measurement_history, state_min, property_min, temperature!, WAAMPrescribedMotion, WAAMHardnessPrescribedMotion, VoxelEnergyFillDynamics, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar, gen_knots, gen_temp_ref, gen_objective_weights, gen_fill_ref, gen_xyz, rollout, marshall_z, QuadraticTrackingObjective, gen_ref, linear2exp, gen_exponential
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
Nkc = 400
Nc = 1
Δtb = 0.02
Δtc = 0.02

Tmin = T₀
Tmax = 3000.0 # K

l = 0.001 # m, aka 1mm

xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)

layers = [[[l; (ny + 1) / 2 * l; 1l], [nx * l; (ny + 1) / 2 * l; 1l]], [[l; (ny + 1) / 2 * l; 1l], [nx * l; (ny + 1) / 2 * l; 1l]]]
p̄ = gen_knots(layers, Nkb, Nkc, Nc)
R = gen_temp_ref(0.1, 1000.0, T∞, A, τ)
a = linear2exp(R, 1000.0, 600.0, Tmax, T∞)
b = gen_exponential(0.1, 1000.0, T∞, A, τ)
@show a
@show b
fill_ref = gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=2.0e-3)#wire_diam)

x̄ = [clamp.(Float64.(f), 0.0, 1.0) for f in fill_ref]

for k in 1:lastindex(fill_ref)
    clamp!(fill_ref[k], 0, Nkb) # 0, Nkb
    fill_ref[k][fill_ref[k] .> 0] .= Nkb
end

QRs = gen_objective_weights(fill_ref)
Qxs = [Diagonal(ones(nvox)) for zi in 1:(Nc*(Nkb+Nkc))]

process = WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄)
process2 = WAAMHardnessPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ)

xg, Qs = gen_ref(fill_ref, b, Tmax, 1000.0, 600.0, T∞, Δtb, ρ, l, cₚ) #R
xf = [(l^3 * ρ * cₚ) * (T∞ + 1) * x̄[end]; x̄[end]]
Qf = 1e4 * Diagonal([1e-2 * ones(nvox); 1e0 * ones(nvox)])
x₀ = zeros(2nvox)
t = Δtb * range(1, Nc * (Nkb + Nkc))

plot(t, [diag(Q)[31] for Q in QRs])
plot(t, [x[31] / (l^3 * ρ * cₚ) for x in xg])
plot(t, [x[nvox+31] for x in xg])

objective = CoolingTrackingObjective(Qxs, QRs, x̄, fill_ref, b, Tmax * (l^3 * ρ * cₚ), T∞ * (l^3 * ρ * cₚ), Qf, xf)
# objective = QuadraticTrackingObjective(Qs, xg)
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=xg[end], Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=false)
# problem2 = AdditiveProblem(process, objective, Nkb, Nkc, Nc, zeros(3nvox), x̄=zeros(3nvox), Δtb=Δtb, Δtc=Δtc, final_constraint=false)

# z₀ = generate_wall_z₀(process2, problem2.idx, zeros(3nvox), Δtb, Δtc; free_time=false)
# X = vcat([[z₀[problem.idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
# U = vcat([[z₀[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)

# animate_3Dstate_history_planar(X, Δtb, nx, ny, nz; path="animation_state3d1_hardness.mp4", strid=2)

function temperature(X, N)
    s = @view X[1:2N]
    T = zeros(N)

    temperature!(process.transfer_dynamics, T, s)
    return T
end

# Y = [temperature(x, nvox) for x in X]
# animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured3d1_hardness.mp4", strid=2)
# Y = [x[(2nvox+1):3nvox] for x in X]
# animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured_frac_3d1_hardness.mp4", strid=2, quantity="Fraction Transformed", scale=(0, 1))

#3000.0
X0 = [[[(l^3 * ρ * cₚ) * 1000.0 * clamp.(x .- 0.4, 0.0, 1.0); clamp.(x .- 0.4, 0.0, 1.0)] for x in x̄]]
U0 = [0.030 * ones(Nkb)]
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc; free_time=true)

z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, c_tol=1.0e-5, xg=x₀, ug=[0.030], z₀=z0)
Y = [temperature(x, nvox) for x in X]

animate_3Dstate_history_planar(X, Δtb, nx, ny, nz; path="animation_state3d8_hardness_optimized.mp4", strid=2)
animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured3d8_hardness_optimized.mp4", strid=2)

t = cumsum(Δt)
save_object("traj_waam3d8_hardness_prescribed.jld2", z)

# X = rollout(process2, zeros(3nvox), [U,], Nkb, Nkc, Nc, Δtb, Δtc, free_time=false)
# X = vcat(X...)
# Y = [x[(2nvox+1):3nvox] for x in X]
# animate_3Dmeasurement_history_planar(Y, X, Δtb, nx, ny, nz; path="animation_measured_frac_3d8_hardness_optimized.mp4", strid=2, quantity="Fraction Transformed", scale=(0, 1))

plot(t, [u[1] for u in U], xlabel="Time (s)", ylabel="Wire Feed Speed (m/s)", label="ȳ=0.1")
# 1: 0.1, 2: 0.4 fixed time
# 3: 0.1, 4: 0.4 free build time
# 5: direct exponential, free time, weights borked
# 6: direct exponential, free time, weights fixed, time bounds borked, weights still probably need adjustment
# 7: adjusted time bounds
# 8: adjusted input bounds

plot(t, [y[46] for y in Y], xlabel="Time (s)", ylabel="Temperature (K)", label="Voxel 46")
plot!(t, clamp.(T̄.(t .- t[Nkb]), 0, Tmax), label="Reference Shifted")

plot(t, [u[1] for u in U] .* (π / 4 * wire_diam^2) ./ (0.0001 ./ Δt))
plot(t, 0.0001 ./ Δt, xlabel="Time (s)", ylabel="Travel Speed (m/s)")
WFS = [u[1] for u in U]
TS = 0.0001 ./ Δt
r = @. wire_diam * √(WFS / (2TS))
plot(t, r, xlabel="Times (s)", ylabel="Bead Radius (m)")