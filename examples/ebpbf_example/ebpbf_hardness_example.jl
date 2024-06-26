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
k = 34.0 # W / mK
ρ = 7826.0 # kg / m^3 # mg
cₚ = 502.416 # J / kg K # mg 
α = k / (ρ * cₚ)
T_AC1 = Inf#900e-3 # kK

# Environment parameters
T∞ = (295.15) * 1e-3 # kK
h∞ = 0.0 # W / m^2 K (Vacuum)

# Geometric parameters
l = 1e-3
lz = 1e-3
stride = 16
ndeep = 3
nsvox = stride * stride
nvox = nsvox * ndeep

# Temporal parameters
Δtb = round(l^2 / α, sigdigits=2) # s
Δtc = Δtb # s
Nc = 1
Nkb = ceil(Int, cₚ * ρ * (lz) * (l * stride)^2 * 700e-3 / (Pₛₑₜ * Δtb))
@show Nkb
Nkc = 50

mask_img = (1.0 .- Float64.(Gray.(load("scotty_bw.png")))) .* 0.8 .+ 0.1
height_original, width_original = size(mask_img)
n_subsample = 7
mask_img_blurred = imfilter(mask_img, Kernel.gaussian(n_subsample))
mask_img_subsampled = mask_img_blurred[4:n_subsample:end, 4:n_subsample:end]
height, width = size(mask_img_subsampled)
mask = vec(mask_img_subsampled)
novox = height * width * ndeep
nosvox = height * width

n_h_strides = ceil(Int, (height - stride) / stride)
n_w_strides = ceil(Int, (width - stride) / stride)
h_strides = [floor(Int, (height - stride) / n_h_strides * k) for k in 0:n_h_strides]
w_strides = [floor(Int, (width - stride) / n_w_strides * k) for k in 0:n_w_strides]
h_windows = [(1+s):(stride+s) for s in h_strides]
w_windows = [(1+s):(stride+s) for s in w_strides]
windows = [(h_range, w_range) for w_range in w_windows for h_range in h_windows]
n_windows = length(windows)

overall_process = PlanarLPBFHardness(height, width, ndeep, l, lz, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, T_AC1, 200.0e-3, Inf * ones(width * height))
OX = [[T∞ * ones(novox); log(-log(1 - 0.1)) * ones(novox)],]
OU = [zeros(nosvox),]
Opoints = []

roi_process = PlanarLPBFHardness(stride, stride, ndeep, l, lz, k, ρ, cₚ, T∞, h∞, σ, Pₛₑₜ, Pₛₑₜ, ω, T_AC1, 200.0e-3, Inf * ones(nsvox))
xₙ = roi_process.input_dynamics.xₙ
zₙ = roi_process.input_dynamics.zₙ

# Set up objective
Q = Diagonal([1e-3 * ones(nvox); ones(nsvox); zeros(nvox - nsvox)])
R = Diagonal(zeros(nsvox))
b = zeros(2nvox)
objective = QuadraticObjective(Q, b, R, Q, zeros(2nvox), zeros(nsvox))

# Combine process and objective to set up problem
problem = AdditiveProblem(roi_process, objective, Nkb, Nkc, Nc, zeros(2nvox), Δtb=Δtb, Δtc=Δtc)

# for (window_i, window) in enumerate(windows)
window = windows[1]
roi = mask_img[window...]
nrows, ncols = size(roi)

# Set up target state and input
ŷ = zeros(nrows, ncols, ndeep)
ŷ[:, :, 1] .= log.(-log.(1 .- roi))
x̄ = vcat(T∞ * ones(nvox), vec(ŷ))
ū = zeros(nsvox)

# Update objective goal state
objective.x̄ .= x̄

# Get initial conditions, update problem
x₀ = [vec(reshape(OX[end][1:novox], (height, width, ndeep))[window..., :]); vec(reshape(OX[end][(novox+1):end], (height, width, ndeep))[window..., :])]
problem.x₀ .= x₀

# Optimize trajectory
J, z, X, U, Δt, λ, μ_xₗ, μ_xᵤ, μ_uₗ, μ_uᵤ = optimize_trajectory(problem; max_iter=1_000, xg=[600e-3 * ones(nvox); log(-log(1 - 0.6)) * ones(nvox)], solv="ma97")
# save_object("traj_lpbf_ebpbf_$(run_num)_opt.jld2", (z, X, U, Δt, λ))
GC.gc()

t = cumsum(Δt) .- Δt[1]

Δtm = 1e-6
simsub = 20
Δtsim = Δtm / simsub

# Calculate spot sequence
UP, points, Dt, dt = field_to_spots(t[1:Nkb] .+ Δtb, (@view U[1:Nkb]), Δtm, Pₛₑₜ, ω, xₙ, zₙ, σ, l; method=:greedy)
Ncool = Nkc * Int(round(Δtb / Δtm))
append!(Dt, Δtm * ones(Ncool))
append!(UP, [zeros(length(UP[1])) for k in 1:Ncool])
tm = cumsum(Dt) .- Dt[1]
t_sim = collect(range(0, tm[end], step=Δtsim))
UPsim = resample_vector_traj(tm, UP, t_sim)
OUPsim = [zeros(nosvox) for _ in UPsim]
for (k, u) in enumerate(UPsim)
    reshape(OUPsim[k], (height, width))[window...] .= u
end

# Simulate spot sequence over the whole part
XPsim = rollout(overall_process, OX[end], OUPsim, length(OUPsim), Δtsim)

# Subsample
t_sim = t_sim[1:10simsub:end]
UPsim = UPsim[1:10simsub:end]
XPsim = XPsim[1:10simsub:end]

# Append to overall simulation
append!(OX, XPsim)
append!(OU, UPsim)
append!(Opoints, points)
append!(Odt, dt)

GC.gc()
# end

# p0 = [0; 0]

# CSV.write("scan_strat_scotty.csv", Tables.table(vcat([[round(x[1], digits=9); round(x[2], digits=9); round(Δt, digits=9)]' for (x, Δt) in zip(Opoints, Odt)]...); header=["X", "Y", "Δt"]))