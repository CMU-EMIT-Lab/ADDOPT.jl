include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, optimize_trajectory, animate_state_history, animate_measurement_history, PlanarLPBF, marshall_z, constraints!, QuadraticObjective, step_RK4, combined_dynamics!
using Plots
using JLD2
using CSV, Tables
using StatsFuns
using Statistics

run_num = 25
η = 1.0
γ = 1.0
if length(ARGS) ≥ 1
    global γ = parse(Float64, ARGS[1]) / 10.0
    global η = parse(Float64, ARGS[2]) / 10.0
    global run_num = "$(γ)_$(η)"
end

# Machine parameters
σ = 0.25e-3 / 2.355 # Spot diameter
vₘₐₓ = 1e3 # m/s
Pₛₑₜ = 3000.0e-3 # W # kW

# Material parameters
k(T) = 42.0 * γ # W / mK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0 # J / kg K 

τ = 1e5#4 # 1/s, fusion rate
Tₛ = (1385.0 + 273.15) * 1e-3 # K, solidus # kK
Tₗ = (1450.0 + 273.15) * 1e-3 # K, liquidus #kK
Tₗ = 1.0
Tₘ = (Tₛ + Tₗ) / 2 # K, melting 
Tboil = 3000.0e-3 # K, boiling # kK

# Geometric parameters
n = 20
nrows = n
ncols = n
nvox = nrows * ncols
l = 5e-3 / n # m, aka 5mm / n 

# Environment parameters
T∞ = 293.15e-3 # kK
h∞ = k(0) * 4 / (√(π) * l * n) * η # W / m^2 K

process = PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, vₘₐₓ, Tₘ, τ, Tboil, 200.0e-3)

T̄ = zeros(nvox)

x₀ = T∞ * ones(nvox)
x̄ = T̄
ū = zeros(nvox)

Q = (I - 1/nvox * ones(nvox, nvox))' * (I - 1/nvox * ones(nvox, nvox)) / nvox
R = zeros(nvox, nvox)
Qf =  Q#10*
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nkb = 800
Nkc = 2
Nc = 1
Δtb = 20e-6 # s, aka 20μs 
Δtc = 20e-6 # s, aka 20μs

# generate initial guess
px = process.transfer_dynamics.xₙ
pz = process.transfer_dynamics.zₙ
cumulative_heat = T∞ * ones(nvox)
x = copy(cumulative_heat)
P0 = []
X0 = []

xtold = l
ztold = l
idx = 1

Nkb = 0

f!(dx, x, u, t, zi) = combined_dynamics!(dx, x, u, process, t, zi, Δtb)

while minimum(cumulative_heat) ≤ Tₗ #+ 200.0e-3
    # for k in 1:Nkb
    global Nkb += 1
    # global idx = argmin(cumulative_heat) 
    global idx = rand(1:nvox)
    xₜ, zₜ = px[idx], pz[idx]

    σb = l / 2
    Pin = zeros(nvox)
    Pin[idx] = Pₛₑₜ
    # Pin = abs.(randn(nvox))
    # Pin .*=  Pₛₑₜ / norm(Pin)
    # Pin = @. (l^2 / (2π * σb^2)) * exp(-((px - xₜ)^2 + (pz - zₜ)^2) / (2σb^2)) * Pₛₑₜ
    u = copy(Pin)#[Pin; (xₜ - xtold) / Δtb; (zₜ - ztold) / Δtb]
    push!(X0, x)

    global x = step_RK4(f!, x, u, Δtb, 0.0, 1)

    push!(P0, u)
    global xtold = xₜ
    global ztold = zₜ

    global cumulative_heat = x[1:nvox]
end
@show Nkb

push!(X0, step_RK4(f!, X0[end], P0[end], Δtc, 0.0, 1))
for k in 1:(Nkc-1)
    push!(X0, step_RK4(f!, X0[end], zeros(nvox), Δtc, 0.0, 1))
end
X0 = [X0,]
U0 = [P0,]

p = plot([std(x[1:nvox] .* 1e3) for x in X0[1]], label="Initial Guess", xlabel="Time Step", ylabel="σ(T) [K]")

problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=x̄, Δtb=Δtb, Δtc=Δtc, final_constraint=false, hessian=true)

z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc)
z, X, U, Δt, λ = optimize_trajectory(problem; max_iter=10_000, z₀=z0, solv="ma97", isqp=true)#"ma77")
GC.gc()

Y = [x[1:nvox] .* 1e3 for x in X]
animate_measurement_history(Y, Δtb * 1000, nrows, ncols, strid=1, path="animation_temperature_ebpbf_$run_num.mp4", scale=(400, round(Tₗ*1e3, sigdigits=2)))
animate_measurement_history([u[1:nvox] for u in U], Δtb * 1000, nrows, ncols, strid=1, path="animation_power_ebpbf_$run_num.mp4", quantity="Power W", scale=(0, 2Pₛₑₜ/nvox))

save_object("traj_lpbf_ebpbf_$run_num.jld2", z)

# z = load_object("traj_lpbf_ebpbf_$run_num.jld2")
# X = vcat([[z[problem.idx.x[c][k]] for k in 1:Nkb] for c in 1:Nc]...)
# U = vcat([[z[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)

xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

T = [x[1:nvox] for x in X]
Tm = [mean(temp*1e3) for temp in T]
Tσ = [std(temp*1e3) for temp in T]

# will = vcat([[x; z; Δtb]' for (x, z) in zip(xt, zt)]...)
# CSV.write("scan_strat_$run_num.csv", Tables.table(will; header=["X", "Y", "Δt"]))