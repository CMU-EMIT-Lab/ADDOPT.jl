using Plots
using JLD2
using Interpolations
using ProgressMeter

ȳ = 0.8

prefix = "temp/1_600 iter nohess uniform start"
path = "$prefix/input_visual_$(ȳ).mp4"

U = load_object("$prefix/traj_waam_U_$(ȳ).jld2")
t = load_object("$prefix/traj_waam_t_$(ȳ).jld2")
Δt = t .- vcat([0.0], t[1:(end-1)])
t .-= t[1]

strid = 20
step = 20
Δt_sim = 0.02 / step

Δr = 40e-3 / 100
WFS = [u[1] for u in U]
TS = Δr ./ Δt

t_sim = collect(range(0, t[end] + 20.0, step=Δt_sim))
Nk_sim = length(t_sim)

WFS_sim = linear_interpolation(t, WFS, extrapolation_bc=Flat()).(t_sim)
TS_sim = linear_interpolation(t, TS, extrapolation_bc=Flat()).(t_sim)


width = 600
height = 400

p = Progress(Nk_sim ÷ strid)
# Plots.scalefontsizes(1.5)

anim = @animate for k in 1:strid:Nk_sim
    plot(t_sim[1:k], WFS_sim[1:k], xlabel="Time (s)", size=(width, height), fmt=:png, label="Wire Feed Speed (m/s)", xlims=(0, t_sim[end]), ylims=(0, maximum(WFS)), thickness_scaling = 1.5)
    plot!(t_sim[1:k], TS_sim[1:k], label="Travel Speed (m/s)")
    next!(p)
end
# Plots.scalefontsizes(1/1.5)
gif(anim, path, fps=(1 / Δt_sim / strid))