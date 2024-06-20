using LinearAlgebra
using ADDOPT
using Plots
using JLD2
using CSV, Tables
using StatsFuns
using Statistics
using Images
using Random

n_refine = 2
mask_img = (1.0 .- Float64.(Gray.(load("scotty_bw.png")))) .* 0.9
mask_img_fine = refine_grid(mask_img, n_refine)
mask = vec(mask_img)
mask_fine = vec(mask_img_fine)

# Machine parameters
σ = 250e-6 / 1.35 # Spot diameter, m #250e-6
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 400.0e-3 # kW

k = 34.0 # W / mK
ρ = 7826.0 # kg / m^3 # mg
cₚ = 502.416 # J / kg K # mg 

Tₛ = (775) * 1e-3 # kK, solidus 
Tₗ = (1000) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 

# Geometric parameters
ncols, nrows = size(mask_img)
ndeep = 4
nsvox = nrows * ncols
nsvox_fine = nsvox * n_refine^2
nvox = nsvox * ndeep
nvox_fine = nvox * n_refine^2
mask = vcat(mask, zeros(Bool, nsvox * (ndeep - 1)))
mask_fine = vcat(mask_fine, zeros(Bool, nsvox_fine * (ndeep - 1)))
l = 200e-6 # m
lz = 200e-6 # m
ncols_fine = ncols * n_refine
nrows_fine = nrows * n_refine
@show nsvox
# Environment parameters
T∞ = (295.15) * 1e-3 # kK
h∞ = 0.0 # W / m^2 K (Vacuum)

Tmax = Tₛ * ones(nvox)
process = PlanarLPBF(nrows, ncols, ndeep, l, lz, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, Tmax, 200.0e-3, Inf * ones(nsvox))
process_fine = PlanarLPBF(nrows_fine, ncols_fine, ndeep, l / n_refine, lz, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, Tmax, 200.0e-3, Inf * ones(nsvox)


T̄ = zeros(nvox)

x₀ = T∞ * ones(nvox)
x₀_sim = T∞ * ones(nvox * n_refine^2)
x̄ = T̄
ū = zeros(nsvox)

Q = (diagm(mask) - (1 / nₕ) * (mask * mask'))' * (diagm(mask) - (1 / nₕ) * (mask * mask')) / nₕ
R = zeros(nsvox, nsvox)
Qf = Q
b = zeros(nvox)
objective = QuadraticObjective(Q, b, R, Qf, x̄, ū)

Nkc = 20
Nc = 1
Δtb = 100e-6 # s 
Δtc = 100e-6 # s

T_melt = Tₗ + 30.0e-3
u0 = mask[1:nsvox] ./ sum(mask) * process.input_dynamics.Pₘₐₓ
Nkb, U0, X0 = initial_guess(process, mask, x₀, Nkc, Δtb, Δtc, T_melt + 0e-3, u0) #increase superheat requirement
@show Nkb

surface_scaled(X; nsvox=nsvox) = [x[1:nsvox] .* 1e3 for x in X]
animate_measurement_history(surface_scaled(X0[1]), Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_ideal_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500, l=1e3 * l)
animate_measurement_history(surface_scaled(U0[1]), Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_ideal_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500, l=1e3 * l)

boxconstraints = [(1, Nkb, T_melt * mask, Tmax),]
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, xfmin=T_melt * mask, x̄=Tmax, Δtb=Δtb, Δtc=Δtc, final_constraint=false, hessian=true, boxconstraints=boxconstraints)
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc)

J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=10_000, z₀=z0, solv="ma97", isqp=true)
save_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2", (z, X, U, Δt, λ))
GC.gc()

# z, X, U, Δt, λ = load_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2")

t = (cumsum(Δt) .- Δt[1]) .* 1e3
xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ
xₙ_fine = process_fine.input_dynamics.xₙ
zₙ_fine = process_fine.input_dynamics.zₙ

animate_measurement_history(surface_scaled(X), Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_ideal_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500, l=1e3 * l)
animate_measurement_history(surface_scaled(U), Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_ideal_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500, l=1e3 * l)

U0c = U0[1]
Uo = U[1:Nkb]
Δtm = 1e-6
function approximate_and_animate(U, stri; method=:random, Δtm=1e-6)
    U = [vec(refine_grid(reshape(u, (ncols, nrows)), n_refine)) ./ n_refine^2 for u in U]
    simsub = 20
    Δtsim = 1e-6 / simsub
    # UP, xtzt, Dt, dt = field_to_spots((t[1:Nkb] ./ 1e3) .+ Δtb, U, Δtm, Pₛₑₜ, ω, xₙ, zₙ, σ, l; method=method)
    UP, xtzt, Dt, dt = field_to_spots((t[1:Nkb] ./ 1e3) .+ Δtb, U, Δtm, Pₛₑₜ, ω, xₙ_fine, zₙ_fine, σ, l / n_refine; method=method)
    Ncool = Nkc * Int(round(Δtb / Δtm))
    append!(Dt, Δtm * ones(Ncool))
    append!(UP, [zeros(length(UP[1])) for k in 1:Ncool])
    tm = cumsum(Dt) .- Dt[1]
    t_sim = collect(range(0, tm[end], step=Δtsim))
    UPsim = resample_vector_traj(tm, UP, t_sim)
    XPsim = rollout(process_fine, x₀_sim, UPsim, length(UPsim), Δtsim)

    t_sim = t_sim[1:(simsub*10):end]
    UPsim = UPsim[1:(simsub*10):end]
    XPsim = XPsim[1:(simsub*10):end]

    animate_measurement_history(surface_scaled(XPsim, nsvox=nsvox_fine), Δtsim * simsub * 10 * 1e3, nrows_fine, ncols_fine, strid=1, path="animation_$(stri)_$(string(method))_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500, l=1e3 * l / n_refine)
    animate_measurement_history(surface_scaled(UPsim, nsvox=nsvox_fine), Δtsim * simsub * 10 * 1e3, nrows_fine, ncols_fine, strid=1, path="animation_$(stri)_$(string(method))_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, Pₛₑₜ * 1e3), width=500, l=1e3 * l / n_refine)

    return t_sim, UPsim, XPsim, xtzt, dt
end
# 1 - random, no travel; 2 - greedy, no travel; 
# 3 - random, travel; 4 - greedy, travel

# # t_sim_ur, Usim_ur, Xsim_ur, points_ur, dt_ur = approximate_and_animate(U0c, "uniform"; method=:random)
# t_sim_ug, Usim_ug, Xsim_ug, points_ug, dt_ug = approximate_and_animate(U0c, "uniform"; method=:greedy)
# save_object("traj_lpbf_ebpbf_$(run_num)_ug.jld2", (t_sim_ug, Usim_ug, Xsim_ug, points_ug, dt_ug))
t_sim_ug, Usim_ug, Xsim_ug, points_ug, dt_ug = load_object("traj_lpbf_ebpbf_$(run_num)_ug.jld2")
GC.gc()
# # t_sim_or, Usim_or, Xsim_or, points_or, dt_or = approximate_and_animate(Uo, "optimized"; method=:random)
# t_sim_og, Usim_og, Xsim_og, points_og, dt_og = approximate_and_animate(Uo, "optimized"; method=:greedy)
# save_object("traj_lpbf_ebpbf_$(run_num)_og.jld2", (t_sim_og, Usim_og, Xsim_og, points_og, dt_og))
t_sim_og, Usim_og, Xsim_og, points_og, dt_og = load_object("traj_lpbf_ebpbf_$(run_num)_og.jld2")
GC.gc()

Δtsm = 50e-6
function spotmelt_and_animate()
    stri = "spotmelt_random"
    simsub = 20
    Δtsim = 1e-6 / simsub
    order_sm = shuffle(findall(>(0), mask_fine))
    xtzt = [[xₙ_fine[i]; zₙ_fine[i]] for i in order_sm]
    dt = Δtsm * ones(length(xtzt))
    UP, Dt = spots_to_field(xtzt, dt, Pₛₑₜ, ω, xₙ_fine, zₙ_fine, σ, l / n_refine; nf=30)
    Ncool = Nkc * Int(round(Δtb / Δtm))
    append!(Dt, Δtm * ones(Ncool))
    append!(UP, [zeros(length(UP[1])) for k in 1:Ncool])
    tm = cumsum(Dt) .- Dt[1]
    t_sim = collect(range(0, Nkb * Δtb + Nkc * Δtc, step=Δtsim))
    UPsim = resample_vector_traj(tm, UP, t_sim)
    XPsim = rollout(process_fine, x₀_sim, UPsim, length(UPsim), Δtsim)

    t_sim = t_sim[1:(simsub*10):end]
    UPsim = UPsim[1:(simsub*10):end]
    XPsim = XPsim[1:(simsub*10):end]

    animate_measurement_history(surface_scaled(XPsim, nsvox=nsvox_fine), Δtsim * simsub * 10 * 1e3, nrows_fine, ncols_fine, strid=1, path="animation_$(stri)_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500, l=1e3 * l / n_refine)
    animate_measurement_history(surface_scaled(UPsim, nsvox=nsvox_fine), Δtsim * simsub * 10 * 1e3, nrows_fine, ncols_fine, strid=1, path="animation_$(stri)_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, Pₛₑₜ * 1e3), width=500, l=1e3 * l / n_refine)

    return t_sim, UPsim, XPsim, xtzt, dt
end

t_sim_sm, Usim_sm, Xsim_sm, points_sm, dt_sm = spotmelt_and_animate()
GC.gc()

T = [x[1:nvox] for x in X]
Tm = [mean(temp[mask] * 1e3) for temp in T]
Tσ = [std(temp[mask] * 1e3) for temp in T]

T0 = [x for x in X0[1]]
T0m = [mean(x[mask] * 1e3) for x in T0]
T0σ = [std(x[mask] * 1e3) for x in T0]

var_comp = plot(xlabel="Time (ms)", ylabel="Standard Deviation of Temperature (K)", tickfontsize=14, labelfontsize=16, legendfontsize=14, size=(800, 800), widen=true, handlelength=8, grid=false)
plot!(var_comp, x_foreground_color_axis=:black, y_foreground_color_axis=:black)
plot!(var_comp, t, T0σ, linewidth=3, thickness_scaling=1, label="Uniform Power, Ideal", c="lightblue")
# plot!(var_comp, t_sim_ur .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_ur], linewidth=3, thickness_scaling=1, label="Uniform Power, Random Approximation", c="blue", linestyle=:dot)
plot!(var_comp, t_sim_ug .* 1e3, [std(t[mask_fine] .* 1e3) for t in Xsim_ug], linewidth=3, thickness_scaling=1, label="Uniform Power, Approximated", c="darkblue", linestyle=:dot)
plot!(var_comp, t, Tσ, linewidth=3, thickness_scaling=1, label="Optimized Power, Ideal", c="indianred")
# plot!(var_comp, t_sim_or .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_or], linewidth=3, thickness_scaling=1, label="Optimized Power, Random Approximation", c="red", linestyle=:dot)
plot!(var_comp, t_sim_og .* 1e3, [std(t[mask_fine] .* 1e3) for t in Xsim_og], linewidth=3, thickness_scaling=1, label="Optimized Power, Approximated", c="darkred", linestyle=:dot)
plot!(var_comp, t_sim_sm .* 1e3, [std(t[mask_fine] .* 1e3) for t in Xsim_sm], linewidth=3, thickness_scaling=1, label="Random Spot Melting", c="purple", linestyle=:dashdot)

# savefig(var_comp, "var_comp.svg")
savefig(var_comp, "var_comp_$(run_num).png")

var_comp_int = plot(xlabel="Time (ms)", ylabel="Integral of Temperature Variance (K²s)", tickfontsize=14, labelfontsize=16, legendfontsize=14, size=(800, 800), widen=true, handlelength=8, grid=false)
plot!(var_comp_int, x_foreground_color_axis=:black, y_foreground_color_axis=:black)
plot!(var_comp_int, t_sim_sm .* 1e3, cumsum([var(t[mask_fine] .* 1e3) for t in Xsim_sm] .* Δtm * 10), linewidth=3, thickness_scaling=1, label="Random Spot Melting", c="purple", linestyle=:dashdot)
plot!(var_comp_int, t_sim_ug .* 1e3, cumsum([var(t[mask_fine] .* 1e3) for t in Xsim_ug] .* Δtm * 10), linewidth=3, thickness_scaling=1, label="Uniform Power, Approximated", c="darkblue", linestyle=:dot)
plot!(var_comp_int, t, cumsum(T0σ .^ 2 .* Δt), linewidth=3, thickness_scaling=1, label="Uniform Power, Ideal", c="darkblue")
plot!(var_comp_int, t_sim_og .* 1e3, cumsum([var(t[mask_fine] .* 1e3) for t in Xsim_og] .* Δtm * 10), linewidth=3, thickness_scaling=1, label="Optimized Power, Approximated", c="indianred", linestyle=:dot)
plot!(var_comp_int, t, cumsum(Tσ .^ 2 .* Δt), linewidth=3, thickness_scaling=1, label="Optimized Power, Ideal", c="indianred")
savefig(var_comp_int, "var_comp_int.svg")
savefig(var_comp_int, "var_comp_int_$(run_num).png")

function comp_plot(n_div, X0, X, quantity, scale, title)
    plot_init = []
    for d in 1:n_div
        idx = (Nkb * d) ÷ n_div
        x = reshape(X0[idx], (ncols, nrows))'
        h = heatmap(x, aspect_ratio=:equal, colorbar=:none, framestyle=:none, clim=scale, size=(500, 500), title="t = $(round(t[idx], digits=1)) ms", titlefontsize=20)
        push!(plot_init, h)
    end
    plot_opt = []
    for d in 1:n_div
        idx = (Nkb * d) ÷ n_div
        x = reshape(X[idx], (ncols, nrows))'
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

temp_comp = comp_plot(3, surface_scaled(T0), surface_scaled(T), "Temperature (K)", (700, 2100), "Temperature Evolution")
savefig(temp_comp, "temp_comp_$(run_num).svg")
savefig(temp_comp, "temp_comp_$(run_num).png")

power_comp = comp_plot(3, surface_scaled(U0[1] ./ (.4^2)), surface_scaled(U./ (.4^2)), "Power Density (W/mm²)", (0, round(Pₛₑₜ .* 1e3 / nₕ * 2/ (.4^2))), "Power Evolution")
savefig(power_comp, "power_comp_$(run_num).svg")
savefig(power_comp, "power_comp_$(run_num).png")

p0 = [0; 0]

# CSV.write("scan_strat_$(run_num)_uniform_random.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(points_ur, dt_ur)]...); header=["X", "Y", "Δt"]))
CSV.write("scan_strat_$(run_num)_uniform_greedy.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(points_ug, dt_ug)]...); header=["X", "Y", "Δt"]))
# CSV.write("scan_strat_$(run_num)_optimized_random.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(points_or, dt_or)]...); header=["X", "Y", "Δt"]))
CSV.write("scan_strat_$(run_num)_optimized_greedy.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(points_og, dt_og)]...); header=["X", "Y", "Δt"]))

CSV.write("scan_strat_$(run_num)_spotmelt_random.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(points_sm, dt_sm)]...); header=["X", "Y", "Δt"]))

function max_in_arr(vec_of_vecs)
    ret = zeros(eltype(vec_of_vecs[1]), size(vec_of_vecs[1]))

    for vec in vec_of_vecs
        ret .= max.(ret, vec)
    end

    return ret
end

# maxT_ur = max_in_arr(Xsim_ur)
maxT_ug = max_in_arr(Xsim_ug)
# maxT_or = max_in_arr(Xsim_or)
maxT_og = max_in_arr(Xsim_og)
maxT_sm = max_in_arr(Xsim_sm)

# hm_ur = heatmap(reshape(maxT_ur[1:nsvox], (ncols, nrows))' .≥ Tₗ, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
# savefig(hm_ur, "melt_ur.png")
hm_ug = heatmap(reshape(maxT_ug[1:nsvox_fine], (ncols_fine, nrows_fine))' .≥ Tₗ, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_ug, "melt_ug_$(run_num).png")
# hm_or = heatmap(reshape(maxT_or[1:nsvox], (ncols_fine, nrows_fine))' .≥ Tₗ, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
# savefig(hm_or, "melt_or.png")
hm_og = heatmap(reshape(maxT_og[1:nsvox_fine], (ncols_fine, nrows_fine))' .≥ Tₗ, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_og, "melt_og_$(run_num).png")
hm_sm = heatmap(reshape(maxT_sm[1:nsvox_fine], (ncols_fine, nrows_fine))' .≥ Tₗ, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_sm, "melt_sm_$(run_num).png")
GC.gc()