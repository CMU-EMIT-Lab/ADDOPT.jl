using Plots
using JLD2
using Interpolations
using ProgressMeter
using Statistics

ȳ = 0.1

prefix = "."#temp/max final temp nohess 750 iter"
path = "$prefix/input_visual_$(ȳ).mp4"
# path = "$prefix/input_visual_$(ȳ).png"


U = load_object("$prefix/traj_waam_U_$(ȳ).jld2")
t = load_object("$prefix/traj_waam_t_$(ȳ).jld2")
Δt = t .- vcat([0.0], t[1:(end-1)])
t .-= t[1]

Nk = length(Δt)
for k in 3:(Nk-2)
    Δt[k] = median(view(Δt, (k-2):(k+2)))
end

strid = 20
step = 20
Δt_sim = 0.02 / step

Δr = 40e-3 / 100
trim = [u[1] for u in U]
TS = Δr ./ Δt
wire_diam = 0.001143 # m, aka 0.045in
r_bead = wire_diam * sqrt(67.7 / (2 * 5.3))
WFS = 2Δr ./ Δt * (r_bead / wire_diam)^2

WFS[trim .== 0] .= 0
TS[trim .== 0] .= .010

t_sim = collect(range(0, t[end] + 20.0, step=Δt_sim))
Nk_sim = length(t_sim)

WFS_sim = linear_interpolation(t, WFS, extrapolation_bc=Flat()).(t_sim)
TS_sim = linear_interpolation(t, TS, extrapolation_bc=Flat()).(t_sim)
trim_sim = linear_interpolation(t, trim, extrapolation_bc=Flat()).(t_sim)


# width = 800
# height = 500
width = 600
height = 400

# scale = 1.35
# Plots.scalefontsizes(scale)
# p = plot(xlims=(0, t[end]), ylims=(0, 0.150), yticks=0:0.03:0.15, thickness_scaling = scale, xlabel="Time (s)",  size=(width, height), title="Optimized Trajectory for ȳ=$ȳ")
# tp = twinx(p)
# plot!(p, t, WFS, label="Wire Feed Speed (m/s)", legend=:topright, ylabel="Speed (m/s)")
# plot!(p, t, TS, label="Travel Speed (m/s)", legend=:topright)
# plot!(tp, t, trim, label="Trim", xlims=(0, t[end]),color=:purple, legend=:bottomright, ylabel="Trim", ylim=(0, 1.4))
# Plots.scalefontsizes(1/scale)
# savefig(p, path)

pr = Progress(Nk_sim ÷ strid)
# Plots.scalefontsizes(1.5)

anim = @animate for k in 1:strid:Nk_sim
    p = plot(t_sim[1:k], WFS_sim[1:k], xlabel="Time (s)", size=(width, height), label="Wire Feed Speed (m/s)", xlims=(0, t_sim[end]), ylims=(0, maximum(WFS)), thickness_scaling = 1.5, fmt=:png)
    tp = twinx(p)
    plot!(t_sim[1:k], TS_sim[1:k], label="Travel Speed (m/s)", fmt=:png)
    plot!(tp, t_sim[1:k], trim_sim[1:k], label="Trim", color=:purple, legend=:bottomright, ylabel="Trim", ylim=(0, 1.4), xlims=(0, t_sim[end]), fmt=:png)
    next!(pr)
end
# Plots.scalefontsizes(1/1.5)
gif(anim, path, fps=(1 / Δt_sim / strid))