using LinearAlgebra
using ADDOPT
using Plots
using JLD2
using CSV, Tables
using StatsFuns
using Statistics
using Images
using ImageFiltering
using Random

# Machine parameters
σ = 500e-6 / 1.35 # Spot diameter, m
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0e-3 # kW

# Material properties
k = 34.0e-3 # kW / K
ρ = 7826.0 # kg / m^3 
cₚ = 502.416e-3 # kJ / kg K
α = k / (ρ * cₚ)
T_AC1 = 800.0#Inf#900e-3 # kK

# Environment parameters
T∞ = 295.15 #(295.15) * 1e-3 # kK
h∞ = 0.0 # W / m^2 K (Vacuum)

# Geometric parameters
l = 1e-3
lz = 1e-3
ndeep = 3

# Load target hardness image, subsample
mask_img = Gray.(load("scotty_bw.png"))
nrows_original, ncols_original = size(mask_img)
n_subsample = 15
mask_img_blurred = imfilter(mask_img, Kernel.gaussian(n_subsample))
mask_img_subsampled = mask_img_blurred[(n_subsample ÷ 2):n_subsample:end, (n_subsample ÷ 2):n_subsample:end]
mask = (1.0 .- Float64.(mask_img_subsampled)) .* 0.8 .+ 0.1
nrows, ncols = size(mask_img_subsampled)
nvox = nrows * ncols * ndeep
nsvox = nrows * ncols

# Temporal parameters
Δtb = round(l^2 / α, sigdigits=2) / 6 # s
Δtc = Δtb # s
Nc = 1
Nkb = ceil(Int, cₚ * ρ * (lz) * (l * ncols) * (l * nrows) * 700 / (Pₛₑₜ * Δtb))
@show Nkb, Δtb
Nkc = ceil(Int, 1.0 / Δtc)
@show Nkc, Δtc

# Specify initial conditions
x₀ = [T∞ * ones(nvox); log(-log(1 - 0.1)) * ones(nvox)]

# Set up process
process = PlanarLPBFHardness(nrows, ncols, ndeep, l, lz, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, T_AC1, 200.0, Inf * ones(nsvox), 38.005, 0.051590, 240.24)
xₙ = process.input_dynamics.xₙ
zₙ = process.input_dynamics.zₙ

# Set up target state and input
ŷ = zeros(ncols, nrows, ndeep)
ŷ[:, :, 1] .= log.(-log.(1 .- mask'))
x̄ = vcat(T∞ * ones(nvox), vec(ŷ))
ū = zeros(nsvox)

# Set up objective
Q = Diagonal([zeros(nvox); ones(nsvox); zeros(nvox - nsvox)])
R = Diagonal(zeros(nsvox))
b = zeros(2nvox)
objective = QuadraticObjective(Q, b, R, Q, x̄, ū)

# Combine process and objective to set up problem
problem = AdditiveProblem(process, objective, Nkb, Nkc, Nc, x₀, Δtb=Δtb, Δtc=Δtc)

# Optimize trajectory
J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=100, xg=[500. * ones(nvox); log(-log(1 - 0.3)) * ones(nvox)], solv="ma97")
# save_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2", (z, X, U, Δt, λ))
GC.gc()

# t = cumsum(Δt) .- Δt[1]

# Δtm = 1e-6
# simsub = 20
# Δtsim = Δtm / simsub

# # Calculate spot sequence
# UP, points, Dt, dt = field_to_spots(t[1:Nkb] .+ Δtb, (@view U[1:Nkb]), Δtm, Pₛₑₜ, ω, xₙ, zₙ, σ, l; method=:greedy)
# Ncool = Nkc * Int(round(Δtb / Δtm))
# append!(Dt, Δtm * ones(Ncool))
# append!(UP, [zeros(length(UP[1])) for k in 1:Ncool])
# tm = cumsum(Dt) .- Dt[1]
# t_sim = collect(range(0, tm[end], step=Δtsim))
# UPsim = resample_vector_traj(tm, UP, t_sim)
# OUPsim = [zeros(nsvox) for _ in UPsim]
# for (k, u) in enumerate(UPsim)
#     reshape(OUPsim[k], (nrows, ncols))[window...] .= u
# end

# # Simulate spot sequence over the whole part
# XPsim = rollout(overall_process, OX[end], OUPsim, length(OUPsim), Δtsim)

# # Subsample
# t_sim = t_sim[1:10simsub:end]
# UPsim = UPsim[1:10simsub:end]
# XPsim = XPsim[1:10simsub:end]

# # Append to overall simulation
# append!(OX, XPsim)
# append!(OU, UPsim)
# append!(Opoints, points)
# append!(Odt, dt)

# GC.gc()
# # end

# # p0 = [0; 0]

# # CSV.write("scan_strat_scotty.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(Opoints, Odt)]...); header=["X", "Y", "Δt"]))