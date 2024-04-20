using CSV: read
using DataFrames
using Interpolations

include("../../src/sysid.jl")

dts::Vector{Vector{Float64}} = []
Ts::Vector{Vector{Float64}} = []
Δt = 0.2

# TODO: add subsampling, deal with Upitt's repeated time-steps

H₀ = 300.0
H₁ = 500.0
θ₀ = [2000.0; 100.0] #unitless, kJ

df = read("examples/inconel/index.csv", DataFrame; delim=",")
nodes = df[!, "node"]
H̄ = df[!, "vickers_hv1"]

for node in nodes
    df = read("examples/inconel/PROB_$(node).txt", DataFrame; delim=" ", header=["Time (s)"; "Temperature (C)"])
    t = df[!, "Time (s)"]
    T = df[!, "Temperature (C)"]

    # t = collect(0.0:Δt:t[end])
    # T = linear_interpolation(t, T, extrapolation_bc=Flat()).(t)

    dt = t[2:end] .- t[1:(end-1)]
    T = T[2:end] .+ 273.15

    push!(dts, dt)
    push!(Ts, T)
end

prob = SYSID_Problem(dts, Ts, H̄, H₀, H₁)
simulate!(prob, θ₀)
@show [Y[end] for Y in prob.Ys]
@show prob.ȳ

θ = fit_avrami(prob, θ₀)

simulate!(prob, θ)
@show [Y[end] for Y in prob.Ys]
@show prob.ȳ