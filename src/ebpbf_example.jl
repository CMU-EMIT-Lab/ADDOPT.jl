include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, optimize_trajectory, animate_state_history, animate_measurement_history, PlanarLPBF, marshall_z, constraints!, QuadraticObjective, initial_guess, rollout, field_to_spots, resample_vector_traj
using Plots
using JLD2
using CSV, Tables
using StatsFuns
using Statistics
using Images

run_num = 25
mask_img = Bool.(Gray.(load("trial.png")))
mask = vec(mask_img)
nₕ = sum(mask)
η = 1.0
γ = 1.0
if length(ARGS) ≥ 1
    global γ = parse(Float64, ARGS[1]) / 10.0
    global η = parse(Float64, ARGS[2]) / 10.0
    global run_num = "$(γ)_$(η)"
end

# Machine parameters
σ = 250e-6 / 2.355 # Spot diameter, m
vₘₐₓ = 4e3 # m/s
Pₛₑₜ = 3000.0e-3 # kW

# Material parameters
k(T) = 42.0 * γ # W / mK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0 # J / kg K, aka kJ / kg kK 

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 

# Geometric parameters
nrows, ncols = size(mask_img)
n = nrows
nvox = nrows * ncols
l = 200e-6 # m 

# Environment parameters
T∞ = 293.15e-3 # kK
h∞ = k(0) * 4 / (√(π) * l * n) * η # W / m^2 K

Tmax = Tboil * ones(nvox)
Tmax[.!mask] .= Tₛ
process = PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, vₘₐₓ, Tmax, 200.0e-3)

T̄ = zeros(nvox)

x₀ = T∞ * ones(nvox)
x̄ = T̄
ū = zeros(nvox)

Q = (I - 1 / nₕ * (mask * mask'))' * (I - 1 / nₕ * (mask * mask')) / nₕ
R = zeros(nvox, nvox)
Qf = Q
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nkc = 2
Nc = 1
Δtb = 100e-6 # s, aka 100μs 
Δtc = 100e-6 # s, aka 100μs

T_melt = Tₗ + 0.0e-3
Nkb, U0, X0 = initial_guess(process, mask, x₀, Nkc, Δtb, Δtc, T_melt - 50e-3)

problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, xfmin=T_melt * mask, x̄=Tmax, Δtb=Δtb, Δtc=Δtc, final_constraint=true, hessian=true)
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc)

z, X, U, Δt, λ = optimize_trajectory(problem; max_iter=10_000, z₀=z0, solv="ma97", isqp=true)
save_object("traj_lpbf_ebpbf_$run_num.jld2", z)
GC.gc()

# z = load_object("temp/e-beam field optimization/traj_lpbf_ebpbf_25.jld2")
# X = vcat([[z[problem.idx.x[c][k]] for k in 1:Nkb] for c in 1:Nc]...)
# U = vcat([[z[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)

t = (cumsum(Δt) .- Δt[1]) .* 1e3
xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

animate_measurement_history([x[1:nvox] .* 1e3 for x in X0[1]], Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_temperature_ebpbf_$run_num.mp4", scale=(1000, Tboil * 1e3), width=500)
animate_measurement_history([u[1:nvox] .* 1e3 for u in U0[1]], Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500)

animate_measurement_history([x[1:nvox] .* 1e3 for x in X], Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_temperature_ebpbf_$run_num.mp4", scale=(1000, Tboil * 1e3), width=500)
animate_measurement_history([u[1:nvox] .* 1e3 for u in U], Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500)

U0c = U0[1]
Uo = U[1:Nkb]
Δtm = 1e-6
function interpolate_and_animate(U, stri; method=:random)
    simsub = 20
    Δtsim = Δtm / simsub
    UP, xtzt, Dt = field_to_spots(t[1:Nkb] ./ 1e3, U, Δtm, Pₛₑₜ, vₘₐₓ, xₙ, zₙ, σ, l; method=method)
    tm = cumsum(Dt) .- Dt[1]
    t_sim = collect(range(0, tm[end], step=Δtsim))
    UPsim = resample_vector_traj(tm, UP, t_sim)
    XPsim = rollout(process, x₀, UPsim, length(UPsim), Δtsim)

    animate_measurement_history([x[1:nvox] .* 1e3 for x in XPsim], Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_$(string(method))_temperature_ebpbf_$run_num.mp4", scale=(1000, Tboil * 1e3), width=500)
    animate_measurement_history([u[1:nvox] .* 1e3 for u in UPsim], Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_$(string(method))_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, Pₛₑₜ * 1e3), width=500)

    return t_sim, UPsim, XPsim, xtzt
end
# 1 - random, no travel; 2 - greedy, no travel; 
# 3 - random, travel; 4 - greedy, travel

t_sim_ur, Usim_ur, Xsim_ur, points_ur = interpolate_and_animate(U0c, "uniform"; method=:random)
t_sim_ug, Usim_ug, Xsim_ug, points_ug = interpolate_and_animate(U0c, "uniform"; method=:greedy)
t_sim_or, Usim_or, Xsim_or, points_or = interpolate_and_animate(Uo, "optimized"; method=:random)
t_sim_og, Usim_og, Xsim_og, points_og = interpolate_and_animate(Uo, "optimized"; method=:greedy)

T = [x[1:nvox] for x in X]
Tm = [mean(temp[mask] * 1e3) for temp in T]
Tσ = [std(temp[mask] * 1e3) for temp in T]

T0 = [x for x in X0[1]]
T0m = [mean(x[mask] * 1e3) for x in T0]
T0σ = [std(x[mask] * 1e3) for x in T0]

var_comp = plot(xlabel="Time (ms)", ylabel="Standard Deviation of Temperature [K]", tickfontsize=14, labelfontsize=16, legendfontsize=16, size=(900, 900), widen=true)
plot!(var_comp, t, T0σ, linewidth=3, thickness_scaling=1, label="Uniform Power Distribution, Ideal", c="lightblue", linestyle=:dot)
plot!(var_comp, t_sim_ur .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_ur], linewidth=3, thickness_scaling=1, label="Uniform Power with Random Interpolation", c="blue")
plot!(var_comp, t_sim_ug .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_ug], linewidth=3, thickness_scaling=1, label="Uniform Power with Greedy Interpolation", c="darkblue")
plot!(var_comp, t, Tσ, linewidth=3, thickness_scaling=1, label="Optimized Power Distribution, Ideal", c="indianred", linestyle=:dot)
plot!(var_comp, t_sim_or .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_or], linewidth=3, thickness_scaling=1, label="Optimized Power with Random Interpolation", c="red")
plot!(var_comp, t_sim_og .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_og], linewidth=3, thickness_scaling=1, label="Optimized Power with Greedy Interpolation", c="darkred")
savefig(var_comp, "var_comp.svg")
savefig(var_comp, "var_comp.png")

var_comp_int = plot(xlabel="Time (ms)", ylabel="Integral of Temperature Variance [K²s]", tickfontsize=14, labelfontsize=16, legendfontsize=16, size=(900, 900), widen=true)
plot!(var_comp_int, t, cumsum(T0σ.^2 .*Δt), linewidth=3, thickness_scaling=1, label="Uniform Power Distribution, Ideal", c="lightblue", linestyle=:dot)
plot!(var_comp_int, t_sim_ur .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_ur].*Δtm/20), linewidth=3, thickness_scaling=1, label="Uniform Power with Random Interpolation", c="blue")
plot!(var_comp_int, t_sim_ug .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_ug].*Δtm/20), linewidth=3, thickness_scaling=1, label="Uniform Power with Greedy Interpolation", c="darkblue")
plot!(var_comp_int, t, cumsum(Tσ.^2 .*Δt), linewidth=3, thickness_scaling=1, label="Optimized Power Distribution, Ideal", c="indianred", linestyle=:dot)
plot!(var_comp_int, t_sim_or .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_or].*Δtm/20), linewidth=3, thickness_scaling=1, label="Optimized Power with Random Interpolation", c="red")
plot!(var_comp_int, t_sim_og .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_og].*Δtm/20), linewidth=3, thickness_scaling=1, label="Optimized Power with Greedy Interpolation", c="darkred")
savefig(var_comp_int, "var_comp_int.svg")
savefig(var_comp_int, "var_comp_int.png")

function comp_plot(n_div, X0, X, quantity, scale, title)
    plot_init = []
    for d in 1:n_div
        idx = (Nkb * d) ÷ n_div
        x = reshape(X0[idx], (nrows, ncols))' .* 1e3
        h = heatmap(x, aspect_ratio=:equal, colorbar=:none, framestyle=:none, clim=scale, size=(500, 500), title="t = $(round(t[idx], digits=1)) ms", titlefontsize=20)
        push!(plot_init, h)
    end
    plot_opt = []
    for d in 1:n_div
        idx = (Nkb * d) ÷ n_div
        x = reshape(X[idx], (nrows, ncols))' .* 1e3
        h = heatmap(x, aspect_ratio=:equal, colorbar=:none, framestyle=:none, clim=scale, size=(500, 500))
        push!(plot_opt, h)
    end

    h2 = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=scale,
        xlims=(1, 1.1), label="", colorbar_title=quantity,
        framestyle=:none, colorbar_titlefontsize=20, ytickfontsize=16)
    plot_arr = [plot_init; plot_opt; h2]
    l = @layout [grid(2, n_div) a{0.035w}]
    comp = plot(plot_arr..., layout=l, size=(1200, 800),
        plot_title=title, plot_titlefontsize=30,
        link=:all)

    return comp
end

temp_comp = comp_plot(4, T0, T, "Temperature [K]", (1000, 3000), "Temperature Evolution")
savefig(temp_comp, "temp_comp.svg")

power_comp = comp_plot(3, U0c, Uo, "Power [W]", (0, round(Pₛₑₜ .* 1e3 / nₕ * 2)), "Power Evolution")
savefig(power_comp, "power_comp.svg")

CSV.write("scan_strat_$(run_num)_uniform_random.csv", Tables.table(vcat([[x[1]; x[2]; Δtm]' for x in points_ur]...); header=["X", "Y", "Δt"]))
CSV.write("scan_strat_$(run_num)_uniform_greedy.csv", Tables.table(vcat([[x[1]; x[2]; Δtm]' for x in points_ug]...); header=["X", "Y", "Δt"]))
CSV.write("scan_strat_$(run_num)_optimized_random.csv", Tables.table(vcat([[x[1]; x[2]; Δtm]' for x in points_or]...); header=["X", "Y", "Δt"]))
CSV.write("scan_strat_$(run_num)_optimized_greedy.csv", Tables.table(vcat([[x[1]; x[2]; Δtm]' for x in points_og]...); header=["X", "Y", "Δt"]))