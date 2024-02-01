using LinearAlgebra
using ADDOPT
using Plots
using JLD2
using CSV, Tables
using StatsFuns
using Statistics
using Images
using Random

run_num = 29
mask_img = Bool.(Gray.(load("trial.png")))
mask = vec(mask_img)
nₕ = sum(mask)

# Machine parameters
σ = 400e-6 / 2.355 # Spot diameter, m #250e-6
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0e-3 # kW

# Material parameters, taken at the solidus
k = 31.1 # W / mK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0 # J / kg K, aka kJ / kg kK 

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 

# Geometric parameters
ncols, nrows = size(mask_img)
ndeep = 3
nsvox = nrows * ncols
nvox = nsvox * ndeep
mask = vcat(mask, zeros(Bool, nsvox * (ndeep - 1)))
l = 320e-6 # m #200
@show nsvox
# Environment parameters
T∞ = (20.0 + 273.15) * 1e-3 # kK
h∞ = 0.0 # W / m^2 K (Vacuum)

Tmax = Tboil * ones(nvox)
Tmax[.!mask] .= Tₛ - 10e-3
process = PlanarLPBF(nrows, ncols, ndeep, l, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, Tmax, 200.0e-3, Inf * mask[1:nsvox])

T̄ = zeros(nvox)

x₀ = T∞ * ones(nvox)
x̄ = T̄
ū = zeros(nsvox)

Q = (diagm(mask) - (1 / nₕ) * (mask * mask'))' * (diagm(mask) - (1 / nₕ) * (mask * mask')) / nₕ
R = zeros(nsvox, nsvox)
Qf = Q
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nkc = 20
Nc = 1
Δtb = 500e-6 # s, aka 500μs 
Δtc = 500e-6 # s, aka 500μs

T_melt = Tₗ + 0.0e-3
u0 = mask[1:nsvox] ./ sum(mask) * process.input_dynamics.Pₘₐₓ
Nkb, U0, X0 = initial_guess(process, mask, x₀, Nkc, Δtb, Δtc, T_melt + 250e-3, u0) #increase superheat requirement
@show Nkb

surface_scaled(X) = [x[1:nsvox] .* 1e3 for x in X]
animate_measurement_history(surface_scaled(X0[1]), Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_ideal_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500)
animate_measurement_history(surface_scaled(U0[1]), Δtb * 1e3, nrows, ncols, strid=1, path="animation_uniform_ideal_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500)

boxconstraints = [(1, Nkb, (T_melt + 150e-3) * mask, Tmax),]
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, xfmin=(T_melt + 150e-3) * mask, x̄=Tmax, Δtb=Δtb, Δtc=Δtc, final_constraint=false, hessian=true, boxconstraints=boxconstraints)
z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc)

z, X, U, Δt, λ = optimize_trajectory(problem; max_iter=10_000, z₀=z0, solv="ma97", isqp=true)
save_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2", z)
GC.gc()

# z = load_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2")
# X = vcat([[z[problem.idx.x[c][k]] for k in 1:(Nkb+Nkc)] for c in 1:Nc]...)
# U = vcat([[z[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)

t = (cumsum(Δt) .- Δt[1]) .* 1e3
xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

animate_measurement_history(surface_scaled(X), Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_ideal_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500)
animate_measurement_history(surface_scaled(U), Δtb * 1e3, nrows, ncols, strid=1, path="animation_optimized_ideal_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, round(2Pₛₑₜ / nₕ * 1e3)), width=500)

U0c = U0[1]
Uo = U[1:Nkb]
Δtm = 1e-6
function approximate_and_animate(U, stri; method=:random, Δtm=1e-6)
    simsub = 20
    Δtsim = 1e-6 / simsub
    UP, xtzt, Dt, dt = field_to_spots((t[1:Nkb] ./ 1e3) .+ Δtb, U, Δtm, Pₛₑₜ, ω, xₙ, zₙ, σ, l; method=method)
    Ncool = Nkc * Int(round(Δtb / Δtm))
    append!(Dt, Δtm * ones(Ncool))
    append!(UP, [zeros(nsvox) for k in 1:Ncool])
    tm = cumsum(Dt) .- Dt[1]
    t_sim = collect(range(0, tm[end], step=Δtsim))
    UPsim = resample_vector_traj(tm, UP, t_sim)
    XPsim = rollout(process, x₀, UPsim, length(UPsim), Δtsim)

    animate_measurement_history(surface_scaled(XPsim), Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_$(string(method))_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500)
    animate_measurement_history(surface_scaled(UPsim), Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_$(string(method))_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, Pₛₑₜ * 1e3), width=500)

    return t_sim, UPsim, XPsim, xtzt, dt
end
# 1 - random, no travel; 2 - greedy, no travel; 
# 3 - random, travel; 4 - greedy, travel

# t_sim_ur, Usim_ur, Xsim_ur, points_ur, dt_ur = approximate_and_animate(U0c, "uniform"; method=:random)
t_sim_ug, Usim_ug, Xsim_ug, points_ug, dt_ug = approximate_and_animate(U0c, "uniform"; method=:greedy)
# t_sim_or, Usim_or, Xsim_or, points_or, dt_or = approximate_and_animate(Uo, "optimized"; method=:random)
t_sim_og, Usim_og, Xsim_og, points_og, dt_og = approximate_and_animate(Uo, "optimized"; method=:greedy)

Δtsm = 150e-6
function spotmelt_and_animate()
    stri = "spotmelt_random"
    simsub = 20
    Δtsim = 1e-6 / simsub
    order_sm = shuffle(findall(>(0), mask))
    xtzt = [[xₙ[i]; zₙ[i]] for i in order_sm]
    dt = Δtsm * ones(length(xtzt))
    UP, Dt = spots_to_field(xtzt, dt, Pₛₑₜ, ω, xₙ, zₙ, σ, l; nf=30)
    Ncool = Nkc * Int(round(Δtb / Δtm))
    append!(Dt, Δtm * ones(Ncool))
    append!(UP, [zeros(nsvox) for k in 1:Ncool])
    tm = cumsum(Dt) .- Dt[1]
    t_sim = collect(range(0, Nkb * Δtb + Nkc * Δtc, step=Δtsim))
    UPsim = resample_vector_traj(tm, UP, t_sim)
    XPsim = rollout(process, x₀, UPsim, length(UPsim), Δtsim)

    animate_measurement_history(surface_scaled(XPsim), Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_temperature_ebpbf_$run_num.mp4", scale=(700, 2100), width=500)
    animate_measurement_history(surface_scaled(UPsim), Δtsim * 1e3, nrows, ncols, strid=simsub * 10, path="animation_$(stri)_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, Pₛₑₜ * 1e3), width=500)

    return t_sim, UPsim, XPsim, xtzt, dt
end

t_sim_sm, Usim_sm, Xsim_sm, points_sm, dt_sm = spotmelt_and_animate()

T = [x[1:nvox] for x in X]
Tm = [mean(temp[mask] * 1e3) for temp in T]
Tσ = [std(temp[mask] * 1e3) for temp in T]

T0 = [x for x in X0[1]]
T0m = [mean(x[mask] * 1e3) for x in T0]
T0σ = [std(x[mask] * 1e3) for x in T0]

var_comp = plot(xlabel="Time (ms)", ylabel="Standard Deviation of Temperature (K)", tickfontsize=14, labelfontsize=16, legendfontsize=14, size=(800, 800), widen=true, handlelength=8)
plot!(var_comp, t, T0σ, linewidth=3, thickness_scaling=1, label="Uniform Power, Ideal", c="lightblue")
# plot!(var_comp, t_sim_ur[1:2000:end] .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_ur][1:2000:end], linewidth=3, thickness_scaling=1, label="Uniform Power, Random Approximation", c="blue", linestyle=:dot)
plot!(var_comp, t_sim_ug[1:2000:end] .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_ug][1:2000:end], linewidth=3, thickness_scaling=1, label="Uniform Power, Approximated", c="darkblue", linestyle=:dot)
plot!(var_comp, t, Tσ, linewidth=3, thickness_scaling=1, label="Optimized Power, Ideal", c="indianred")
# plot!(var_comp, t_sim_or[1:2000:end] .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_or][1:2000:end], linewidth=3, thickness_scaling=1, label="Optimized Power, Random Approximation", c="red", linestyle=:dot)
plot!(var_comp, t_sim_og[1:2000:end] .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_og][1:2000:end], linewidth=3, thickness_scaling=1, label="Optimized Power, Approximated", c="darkred", linestyle=:dot)
plot!(var_comp, t_sim_sm[1:2000:end] .* 1e3, [std(t[mask] .* 1e3) for t in Xsim_sm][1:2000:end], linewidth=3, thickness_scaling=1, label="Random Spot Melting", c="purple", linestyle=:dashdot)

# savefig(var_comp, "var_comp.svg")
savefig(var_comp, "var_comp_$(run_num).png")

var_comp_int = plot(xlabel="Time (ms)", ylabel="Integral of Temperature Variance (K²s)", tickfontsize=14, labelfontsize=16, legendfontsize=14, size=(800, 800), widen=true, handlelength=8)
plot!(var_comp_int, t, cumsum(T0σ .^ 2 .* Δt), linewidth=3, thickness_scaling=1, label="Uniform Power, Ideal", c="lightblue")
# plot!(var_comp_int, t_sim_ur[1:2000:end] .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_ur] .* Δtm / 20)[1:2000:end], linewidth=3, thickness_scaling=1, label="Uniform Power, Random Approximation", c="blue", linestyle=:dot)
plot!(var_comp_int, t_sim_ug[1:2000:end] .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_ug] .* Δtm / 20)[1:2000:end], linewidth=3, thickness_scaling=1, label="Uniform Power, Approximated", c="darkblue", linestyle=:dot)
plot!(var_comp_int, t, cumsum(Tσ .^ 2 .* Δt), linewidth=3, thickness_scaling=1, label="Optimized Power, Ideal", c="indianred")
# plot!(var_comp_int, t_sim_or[1:2000:end] .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_or] .* Δtm / 20)[1:2000:end], linewidth=3, thickness_scaling=1, label="Optimized Power, Random Approximation", c="red", linestyle=:dot)
plot!(var_comp_int, t_sim_og[1:2000:end] .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_og] .* Δtm / 20)[1:2000:end], linewidth=3, thickness_scaling=1, label="Optimized Power, Approximated", c="darkred", linestyle=:dot)
plot!(var_comp_int, t_sim_sm[1:2000:end] .* 1e3, cumsum([var(t[mask] .* 1e3) for t in Xsim_sm] .* Δtm / 20)[1:2000:end], linewidth=3, thickness_scaling=1, label="Random Spot Melting", c="purple", linestyle=:dashdot)
# savefig(var_comp_int, "var_comp_int.svg")
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

power_comp = comp_plot(3, surface_scaled(U0[1]), surface_scaled(U), "Power (W)", (0, round(Pₛₑₜ .* 1e3 / nₕ * 2)), "Power Evolution")
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

# hm_ur = heatmap(reshape(maxT_ur[1:nsvox], (ncols, nrows))' .≥ T_melt, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
# savefig(hm_ur, "melt_ur.png")
hm_ug = heatmap(reshape(maxT_ug[1:nsvox], (ncols, nrows))' .≥ T_melt, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_ug, "melt_ug.png")
# hm_or = heatmap(reshape(maxT_or[1:nsvox], (ncols, nrows))' .≥ T_melt, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
# savefig(hm_or, "melt_or.png")
hm_og = heatmap(reshape(maxT_og[1:nsvox], (ncols, nrows))' .≥ T_melt, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_og, "melt_og.png")
hm_sm = heatmap(reshape(maxT_sm[1:nsvox], (ncols, nrows))' .≥ T_melt, aspect_ratio=:equal, colorbar=:none, size=(700, 700))
savefig(hm_sm, "melt_sm.png")
GC.gc()