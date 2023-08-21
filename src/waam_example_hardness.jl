include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, QuadraticObjective, TimeWeightedQuadraticObjective, optimize_trajectory, generate_wall_z₀, temperature!, WAAMPrescribedMotion, WAAMHardnessPrescribedMotion, animate_3Dmeasurement_history_planar, animate_3Dstate_history_planar, gen_knots, gen_fill_ref, gen_xyz, rollout, marshall_z, ThermalICProblem, optimize_thermal_ic, resample_vector_traj, WAAMHardnessCooling, constraints!, thermal_ic_to_property_final, gen_torch_ref
using Plots
using JLD2


A = 1e4
τ = 9625
# σ = 5.670374419 * 10^(-8) # Stefan-Boltzmann, W / (m⁴⋅ K⁴)

h∞ = 10 # W / m^2 K
h₀ = 7500 # W / m^2 K
hₐᵣ = 500
η = 0.8

k = 34.0 # W / mK
ρ = 7826.0e6 # kg / m^3 # mg
cₚ = 502.416e-6 # J / kg K # mg 

T∞ = 295.0 # K
T₀ = 295.0 # K
wire_diam = 0.001143 # m, aka 0.045in
Tₗ = 1784.0 # K, liquidus

nx = 25
ny = 4
nz = 4
nvox = nz * ny * nx

Nkb = 100
Nkc = 200
Nc = 3
Δtb = 0.02
Δtc = 0.02

Tmin = T₀
Tmax = 3000.0 # K

l = 0.002 # m, aka 2mm
r_bead = wire_diam * sqrt(67.7 / (2 * 5.3))

xₙ, yₙ, zₙ = gen_xyz(nx, ny, nz, l)

layers = [[[3l; (ny + 1) / 2 * l; 1l - l / 2], [(nx - 2) * l; (ny + 1) / 2 * l; 1l - l / 2]],
    [[(nx - 2) * l; (ny + 1) / 2 * l; 2l - l / 2], [3l; (ny + 1) / 2 * l; 2l - l / 2]],
    [[3l; (ny + 1) / 2 * l; 3l - l / 2], [(nx - 2) * l; (ny + 1) / 2 * l; 3l - l / 2]]]
p̄ = gen_knots(layers, Nkb, Nkc, Nc)
fill_ref = gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=r_bead, l=l)
torch_ref = gen_torch_ref(p̄, xₙ, yₙ, zₙ; radius=r_bead, l=l)
for c in 1:Nc
    for k in (Nkb+1):(Nkb+Nkc)
        zi = (c - 1) * (Nkb + Nkc) + k
        torch_ref[zi] .= 0
    end
end

x̄ = copy(fill_ref)
display(reshape(x̄[end], (nx, ny, nz)))
Δr = norm(p̄[2] .- p̄[1]) * ones(Nc*(Nkb+Nkc))

process = WAAMPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, r_bead, Δr, fill_ref, torch_ref)

function temperature(X, N)
    s = @view X[1:N]
    T = zeros(N)

    temperature!(process.transfer_dynamics, T, s)
    return T
end

ȳ = 0.8
if length(ARGS) ≥ 1
    global ȳ = parse(Float64, ARGS[1])
end
xref = copy(x̄[end])
Nkic = 1000
Qic = 10.0 * Diagonal(ones(nvox))
Eₘᵢₙ = 1000.0 * (l^3 * ρ * cₚ) * xref
Eₘₐₓ = Tₗ * (l^3 * ρ * cₚ) * xref
J = zeros(nvox, nvox)
display(reshape(xref, (nx, ny, nz)))
cooling_process = WAAMHardnessCooling(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄, A, τ, [xref for k in 1:Nkic])
icprob = ThermalICProblem(cooling_process, nvox, xref, zeros(nvox), Eₘᵢₙ, Eₘₐₓ, Δtc, Nkic, ȳ * ceil.(xref), Qic, J)
Ei = nothing
if isfile("ic_waam_Ei_$(ȳ).jld2")
    global Ei = load_object("ic_waam_Ei_$(ȳ).jld2")
else
    global Ei = optimize_thermal_ic(icprob; tol=1e-6, c_tol=1e-6, max_iter=150)
    save_object("ic_waam_Ei_$(ȳ).jld2", Ei)
end
yg = thermal_ic_to_property_final(icprob, Ei)
Ei .= Ei ./ xref
map!(x -> isnan(x) ? 0.0 : x, Ei, Ei)
display(reshape(yg, (nx, ny, nz)))
display(reshape(Ei ./ (ρ * cₚ * l^3), (nx, ny, nz)))

Q = [Diagonal(10 * exp(-2 * (zi / (Nkb + Nkc))) * (ones(nvox) .- 0.99 * ceil.(tr))) for (zi, tr) in zip(((Nkb+Nkc)*Nc):-1:1, torch_ref)]
x₀ = x̄[1] .* (ρ * cₚ * l^3 * T∞)
xg = [Ei .* x for x in x̄]
R = Diagonal([5e-2])
ū = [1.0]
Δt̄b = 0.076
objective = TimeWeightedQuadraticObjective(Q, R, xg, ū, Δt̄b)
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=xg[end], Δtb=nothing, Δtc=nothing, final_constraint=false, hessian=false,
    Δtb_min=0.04, Δtb_max=0.12,
    Δtc_min=0.005, Δtc_max=0.10)

X0 = [[(l^3 * ρ * cₚ) * (600.0 + (Tₗ - 600.0) * (i < Nkb ? i / Nkb : (Nkc - (i - Nkb)) / Nkc)) * x for (i, x) in enumerate(x̄[((c-1)*(Nkb+Nkc)+1):(c*(Nkb+Nkc))])] for c in 1:Nc]
U0 = [[[1.0] for k in 1:Nkb] for c in 1:Nc]
# X0 = rollout(process, x₀, U0, Nkb, Nkc, Nc, 0.075, 0.075)
# X0 = [[x .+ rand(2nvox)./10 for x in X] for X in X0]
z0 = marshall_z(problem.idx, X0, U0, 0.075, 0.08; free_time=true)

if isfile("traj_waam_z_$(ȳ).jld2")
    global z0 = load_object("traj_waam_z_$(ȳ).jld2")
end

# z, X, U, Δt = optimize_trajectory(problem; max_iter=900, tol=1e-5, c_tol=1.0e-5, z₀=z0, solv="ma97")
# t = cumsum(Δt)
# save_object("traj_waam_z_$(ȳ).jld2", z)
# save_object("traj_waam_X_$(ȳ).jld2", X)
# save_object("traj_waam_U_$(ȳ).jld2", U)
# save_object("traj_waam_t_$(ȳ).jld2", t)

z = load_object("traj_waam_z_$(ȳ).jld2")
X = load_object("traj_waam_X_$(ȳ).jld2")
U = load_object("traj_waam_U_$(ȳ).jld2")
t = load_object("traj_waam_t_$(ȳ).jld2")
Δt = t .- vcat([0.0], t[1:(end-1)])
t .-= t[1]

slow_fac = 20
Δt_sim = Δtb / slow_fac
t_sim = collect(range(0, t[end] + 20.0, step=Δt_sim))
Nk_sim = length(t_sim)

p̄_sim = resample_vector_traj(t, p̄, t_sim)
fill_ref_sim = resample_vector_traj(t, fill_ref, t_sim)
torch_ref_sim = resample_vector_traj(t, torch_ref, t_sim)
Δr_sim = [norm(p̄_sim[k+1] .- p̄_sim[k]) for k in 1:(Nk_sim-1)] ## BETERR, still wrong (simulation is constant time increment rather than constant dist increment)
push!(Δr_sim, Δr_sim[end])
process2 = WAAMHardnessPrescribedMotion(nx, ny, nz, l, k, ρ, cₚ, T∞, T₀, wire_diam, h∞, h₀, hₐᵣ, η, Tmin, Tmax, p̄_sim, A, τ, r_bead, Δr_sim, fill_ref_sim, torch_ref_sim)

U_sim = resample_vector_traj(t, U, t_sim)
X_opt = resample_vector_traj(t, X, t_sim)
Y_temp_opt = [temperature(x, nvox) for x in X_opt]
# animate_3Dmeasurement_history_planar(Y_temp_opt, fill_ref_sim, Δt_sim, nx, ny, nz; path="animation_waam_temperature_$(ȳ)_opt.mp4", strid=slow_fac, l=2)
# Append for final cooling, to time and input

X_sim = rollout(process2, [x₀; zeros(nvox)], U_sim, Nk_sim, Δt_sim)
# animate_3Dstate_history_planar(X_sim, Δt_sim, nx, ny, nz; path="animation_waam_states_$(ȳ).mp4", strid=slow_fac, l=2)
Y_temp = [temperature(x, nvox) for x in X_sim]
# animate_3Dmeasurement_history_planar(Y_temp, fill_ref_sim, Δt_sim, nx, ny, nz; path="animation_waam_temperature_$(ȳ).mp4", strid=slow_fac, l=2)
Y_frac = [x[(nvox+1):2nvox] for x in X_sim]
# animate_3Dmeasurement_history_planar(Y_frac, fill_ref_sim, Δt_sim, nx, ny, nz; path="animation_waam_fraction_$(ȳ).mp4", strid=slow_fac, quantity="Fraction Transformed", scale=(0, 1), l=2)

TS = Δr ./ Δt
plot(t, TS, xlabel="Times (s)", ylabel="Travel Speed (m/s)")
savefig("fig_waam_ts_$(ȳ).png")

WFS = 2 .* TS .* (r_bead / wire_diam)^2
plot(t, WFS, xlabel="Time (s)", ylabel="Wire Feed Speed (m/s)", label="ȳ=$(ȳ)")
savefig("fig_waam_wfs_$(ȳ).png")

r = @. wire_diam * √(WFS / (2TS))
plot(t, r, xlabel="Times (s)", ylabel="Bead Radius (m)")
savefig("fig_waam_bead_diam_$(ȳ).png")

plot(t, Δt, xlabel="Times (s)", ylabel="Time Step (s)")
savefig("fig_waam_timestep_$(ȳ).png")

trim = [u[1] for u in U]
plot(t, trim, xlabel="Time (s)", ylabel="Trim")
savefig("fig_waam_trim_$(ȳ).png")