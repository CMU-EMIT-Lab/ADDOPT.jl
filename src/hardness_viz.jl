using Plots
using CSV
using DataFrames
using Statistics
using HypothesisTests
using StatsFuns

data_08 = CSV.read("WAAM Beads - Test job # 11.tsv", DataFrame; delim=" ", header=["HV"; "X (mm)"; "Y (mm)"])
data_01 = CSV.read("WAAM Beads - Test job # 12.tsv", DataFrame; delim=" ", header=["HV"; "X (mm)"; "Y (mm)"])


hv_01 = data_01[!, "HV"]
x_01 = data_01[!, "X (mm)"]
y_01 = data_01[!, "Y (mm)"]

y_01 .*= -1

hv_08 = data_08[!, "HV"]
x_08 = data_08[!, "X (mm)"]
y_08 = data_08[!, "Y (mm)"]

y_08 .*= -1

hvsubs_01 = hv_01[y_01.>0.5]
@show mean(hvsubs_01)
@show std(hvsubs_01)
@show std(hvsubs_01) / √(length(hvsubs_01))

hvsubs_08 = hv_08[y_08.>0.5]
@show mean(hvsubs_08)
@show std(hvsubs_08)
@show std(hvsubs_08) / √(length(hvsubs_08))

t = (mean(hvsubs_08) - mean(hvsubs_01)) / √(var(hvsubs_08) / length(hvsubs_08) + var(hvsubs_01) / length(hvsubs_01))
dof = (var(hvsubs_08)^2 / length(hvsubs_08) + var(hvsubs_01)^2 / length(hvsubs_01))^2 / ((var(hvsubs_08) / length(hvsubs_08))^2 / (length(hvsubs_08) - 1) + (var(hvsubs_01) / length(hvsubs_01))^2 / (length(hvsubs_01) - 1))

@show t
@show dof

s08 = scatter(x_08, y_08, zcolor=hv_08, markersize=8, clim=(140, 250), aspect_ratio=:equal, label=nothing, title="ȳ=0.8")
s01 = scatter(x_01, y_01, zcolor=hv_01, markersize=8, clim=(140, 250), aspect_ratio=:equal, label=nothing, title="ȳ=0.1")

UnequalVarianceTTest(hvsubs_01, hvsubs_08)

plot(s08, s01, layout=(1, 2))

p = plot(title="Comparison of Measured Hardness", xlabel="HV", ylabel="Frequency", grid=nothing, #size=(900, 600),
        titlefontsize=17,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=11)
histogram!(p, hvsubs_08, fillopacity=0.5, label="ȳ=0.8, Data", normalize=true, linewidth=0, color=:blue)
histogram!(p, hvsubs_01, fillopacity=0.5, label="ȳ=0.1, Data", normalize=true, linewidth=0, color=:red)

plot!(p, x -> normpdf(mean(hvsubs_08), std(hvsubs_08), x), label="ȳ=0.8, Fit", linewidth=3, color=:blue)
vline!(p, [mean(hvsubs_08)], color=:blue, label=nothing, linewidth=2, linestyle=:dash)

plot!(p, x -> normpdf(mean(hvsubs_01), std(hvsubs_01), x), label="ȳ=0.1, Fit", linewidth=3, color=:red)
vline!(p, [mean(hvsubs_01)], color=:red, label=nothing, linewidth=2, linestyle=:dash)

display(p)
