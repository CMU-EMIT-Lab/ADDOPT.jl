include("ADDOPT.jl")
using LinearAlgebra
using .ADDOPT: AdditiveProblem, QuadraticCubicObjective, optimize_trajectory, animate_state_history, animate_measurement_history, PlanarLPBF, marshall_z, constraints!, QuadraticObjective
using Plots
using JLD2
using CSV, Tables
using StatsFuns

# Machine parameters
σ = 0.25e-3 / 2.355#(0.25 / 2.355) * 1e-3 # m 50
vₘₐₓ = 1e3 # m/s
Pₛₑₜ = 3000.0e-3 # W # kW

# Environment parameters
T∞ = 873.15e-3 # K #kK
h∞ = 75.0 # W / m^2 K
# h₀ = 300.0 # W / m^2 K

# Material parameters
k(T) = 30.5 # W / mK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0 # J / kg K 

τ = 1e5#4 # 1/s, fusion rate
Tₛ = (1385.0 + 273.15) * 1e-3 # K, solidus # kK
Tₗ = (1450.0 + 273.15) * 1e-3 # K, liquidus #kK
Tₘ = (Tₛ + Tₗ) / 2 # K, melting 
Tboil = 3000.0e-3 # K, boiling # kK

# Geometric parameters
n = 40
nrows = n
ncols = n
nvox = nrows * ncols
l = 10e-3 / n # m, aka 10mm / n 

process = PlanarLPBF(nrows, ncols, l, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, vₘₐₓ, Tₘ, τ, Tboil, 200.0e-3)

# Q = 1e-0 * Diagonal(ones(nvox)) #100
# C = 1e-3 * ones(nvox)

T̄ = T∞ * ones(nvox)#Tₗ * ones(nvox)#(Tₗ + 80.0e-3) * ones(nvox)
ȳ = Tₗ * ones(nvox)

# x₀ = [T∞ * ones(nvox); T∞ * ones(nvox); l; l] # zeros(nvox)
# x̄ = [T̄; ȳ; l; l]
x₀ = [T∞ * ones(nvox); l; l] # zeros(nvox)
x̄ = [T̄; l; l]
ū = zeros(nvox + 2)

# objective = QuadraticCubicObjective(nvox, C, Q, T̄, ȳ)
# Q = Diagonal([1e-1 * ones(nvox); 1e2 * ones(nvox); 0.0; 0.0])
Q = Diagonal([1e0 * ones(nvox); 0.0; 0.0])
R = Diagonal(zeros(nvox + 2))
Qf = 10 * Q
objective = QuadraticObjective(Q, R, Qf, x̄, ū)

Nkb = 800
Nkc = 2
Nc = 1
Δtb = 25e-6 # s, aka 25μs 
Δtc = 25e-6 # s, aka 25μs

# X0 = [[[(k / (Nkb + Nkc) * (Tₗ + 400.0 - T∞) - T∞) * ones(nvox); k / (Nkb + Nkc) * ones(nvox); n * l * rand(2)] for k in 1:(Nkb+Nkc)] for c in 1:Nc]
# U0 = [[[Pₛₑₜ / nvox * rand(nvox); vₘₐₓ / 5 * randn(2)] for k in 1:Nkb] for c in 1:Nc]

# generate initial guess
px = process.transfer_dynamics.xₙ
pz = process.transfer_dynamics.zₙ
cumulative_heat = T∞ * ones(nvox)
cumulative_fuse = zeros(nvox)
P0 = []
X0 = []


xtold = l
ztold = l
idx = 1

Nkb = 0
while minimum(cumulative_fuse) ≤ Tₗ
    # for k in 1:Nkb
    global Nkb += 1
    global idx = argmin(cumulative_heat)
    xₜ, zₜ = px[idx], pz[idx]

    σb = l / 2#.5
    Pin = @. (l^2 / (2π * σb^2)) * exp(-((px - xₜ)^2 + (pz - zₜ)^2) / (2σb^2)) * Pₛₑₜ

    cumulative_fuse .= max.(cumulative_fuse, cumulative_heat)#logistic.((cumulative_heat.-Tₘ)))

    push!(X0, [cumulative_heat; xₜ; zₜ])
    push!(P0, [Pin; (xₜ - xtold) / Δtb; (zₜ - ztold) / Δtb])
    global xtold = xₜ
    global ztold = zₜ

    cumulative_heat .+= Pin / (l^3 * ρ * cₚ) * Δtb
end
@show Nkb

for k in 1:Nkc
    push!(X0, X0[end])
end
X0 = [X0,]
U0 = [P0,]

# println(X0[1][end])

problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, x̄=x̄, Δtb=Δtb, Δtc=Δtc, final_constraint=false, hessian=true)

z0 = marshall_z(problem.idx, X0, U0, Δtb, Δtc)
z, X, U, Δt = optimize_trajectory(problem; max_iter=10_000, z₀=z0, solv="ma77")

Y = [x[1:nvox] .* 1e3 for x in X]
animate_measurement_history(Y, Δtb * 1000, nrows, ncols, strid=1, path="animation_temperature_ebpbf_10.mp4", scale=(600, 2200))
animate_measurement_history([u[1:nvox] .* 1e3 for u in U], Δtb * 1000, nrows, ncols, strid=1, path="animation_power_ebpbf_10.mp4", quantity="Power W", scale=(0, 4000))

save_object("traj_lpbf_ebpbf_10.jld2", z)

# z = load_object("traj_lpbf_ebpbf_10.jld2")
X = vcat([[z[problem.idx.x[c][k]] for k in 1:Nkb] for c in 1:Nc]...)
U = vcat([[z[problem.idx.u[c][k]] for k in 1:Nkb] for c in 1:Nc]...)

xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

xt = [x[end-1] for x in X]
zt = [x[end] for x in X]

P = [sum(u[1:nvox]) for u in U]
plot(P)

plot(xt, zt)
savefig("scan_strat_10.png")

will = vcat([[x; z; Δtb]' for (x, z) in zip(xt, zt)]...)
CSV.write("scan_strat_10.csv", Tables.table(will; header=["X", "Y", "Δt"]))

# 1 - sigmoid fusion dynamics, no minimum power, maximum number of iterations exceeded
# 2 - max temp dynamics, maxed out on iterations
# 3 - direct quadratic temperatue objective 500 um, 10x10
# 4 - direct quadratic temperatue objective 350 um, 10x10
# 5 - direct quadratic temperatue objective 250 um, 10x10
# 6 - direct quadratic temperatue objective, single voxel constraint, 10x10, 1.169e4 solution
# 7 - direct quadratic temperatue objective, low T goal, single voxel constraint, 10x10, 1.172e4 solution with T\infty as goal 5mmx5mm, 36s
# 8 - direct quadratic temperatue objective, low T goal, single voxel constraint, 20x20 with 5mmx5mm, 200s
# 8 - direct quadratic temperatue objective, low T goal, single voxel constraint, 40x40 with 10mmx10mm