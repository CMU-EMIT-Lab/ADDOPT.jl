using ADDOPT
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using CSV, Tables
using Printf
using JLD2

HVmin = 135.0
HVmax = 404.0
y_init = (HVmax - 365) / (HVmax - HVmin)

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

# Geometric parameters
mask_img = Ty.(Gray.(load("examples/ebpbf/scotty_bw.png")))

n_subsample = 5
mask_img_blurred = imfilter(mask_img, Kernel.gaussian(n_subsample))
mask_img_subsampled = mask_img_blurred[(n_subsample÷2):n_subsample:end, (n_subsample÷2):n_subsample:end]
mask_img_subsampled = (1.0 .- Float64.(mask_img_subsampled)) .* (0.9 - y_init) .+ y_init

mask_top = vec(mask_img_subsampled)
Nx, Ny = size(mask_img_subsampled)
Nz = 4
buffer = 2
nsvox = Ny * Nx
nvox = nsvox * Nz
Nu = (Nx - 2buffer) * (Ny - 2buffer)
mask = vcat(mask_top, y_init * ones(Ty, nsvox * (Nz - 1)))

l = 1e-3 * n_subsample / 7 # m

@show Nx, Ny
@show nsvox
@show nvox

η = 1.0
σ = 1200e-6 / 1.35 # Spot diameter, m
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0 * η # W

# Material parameters, taken at the solidus
k = 31.1e3 # W / kK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0e3 # J / kg kK 

# Tempering parameters
lnA::Ty = 8.845 # 38.005
n::Ty = 0.358 # 0.051590
E::Ty = 56.277e-3 # 240.24 # kJ / mol K

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 
T_AC1 = 1000.0e-3 # kK

# Environment parameters
T∞ = (20.0 + Pₛₑₜ * 0.100 * 50 / ((1 / 2) * (0.1)^2 * sqrt(k / ρ / cₚ * 5.0) * ρ * cₚ) + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)

dt::Ty = 50e-3 # s, aka 50ms

Tmax = T_AC1 * ones(nvox)

tempering = Tempering(lnA, n, E, dt; num = nvox)
pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; σ = σ, buffer = buffer)
dynamics = PBFTempering(pbf_powerfield, tempering)

power_mask = zeros(Bool, (Nx, Ny, Nz))
power_mask[(1+buffer):(end-buffer), (1+buffer):(end-buffer), 1] .= true
power_mask = vec(power_mask)

x₀ = V([T∞ * ones(nvox); log(-log(1 - y_init)) * ones(nvox)])
uw = (mask_top[power_mask[1:nsvox]] .+ 0.2) * Pₛₑₜ / sum((mask_top[power_mask[1:nsvox]] .+ 0.2))
uw = V(uw)
u0 = zeros(Nu)
u0 = V(u0)

Nk = 0
Nkc = 0
x1, x2 = copy(x₀), copy(x₀)
for i in 1:1000
    transition!(dynamics, x2, x1, uw)
    if maximum(x2) ≥ T_AC1 - 100e-3
        break
    end
    global Nk = i
    x1 .= x2
end
for i in 1:1000
    transition!(dynamics, x2, x1, u0)
    if maximum(x2) ≤ 450e-3
        break
    end
    global Nkc = i
    x1 .= x2
end
@show Nk = Nk + Nkc

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, 2nvox)
b_x_eq = V(undef, 0)
A_u_eq = M(ones(1, Nu) ./ Pₛₑₜ)
b_u_eq = V([1.0])
A_x_ineq = M([collect(1.0 * I(nvox)) zeros(nvox, nvox)])
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(Nu)))
b_u_ineq = V(zeros(Nu))

build_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
cooling_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_ineq, b_u_ineq, A_x_ineq, b_x_ineq, M(undef, 0, Nu), V(undef, 0))
constraints = vcat([build_constraint for _ in 1:(Nk-Nkc)], [cooling_constraint for _ in 1:Nkc])

x̄ = V([T∞ * ones(nvox); log.(-log.(1 .- mask))])
ū = V(zeros(Nu))

Q = diagm([zeros(nvox); 1e-2 / Nk / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
Qf = diagm([zeros(nvox); 1e3 / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
R = diagm(zeros(Nu))
Q = M(Q)
Qf = M(Qf)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)
costs = [QuadraticCost(Qf .* Ty(exp(-(Nk - k) / 5)), R, x̄, ū) for k in 1:(Nk-1)]
final_cost = QuadraticCost(Qf, R, x̄, ū)
push!(costs, final_cost)

problem = Problem(x₀, process, costs, constraints, M)

N_runs = 159

optim_costs = zeros(N_runs)
init_costs = zeros(N_runs)
for iter in 1:N_runs
    U0 = load_object("output/EB-PBF - Hardness/attractor_experiment/initialguess_U_$(iter)_$(n_subsample).jld2")
    U0 = [V(u) for u in U0]
    rollout!(problem, U0)
    init_costs[iter] = eval_cost(problem)

    U = load_object("output/EB-PBF - Hardness/attractor_experiment/solution_U_$(iter)_$(n_subsample).jld2")
    U = [V(u) for u in U]

    rollout!(problem, U)
    optim_costs[iter] = eval_cost(problem)
end

histogram(init_costs, label = "Initial")
histogram!(optim_costs, label = "Optimized")
savefig("guess_to_optim_hist.svg")
savefig("guess_to_optim_hist.png")

X = hcat(init_costs, ones(N_runs))
Y = optim_costs
Θ = X \ Y
f = X * Θ
ss_res = mapreduce((y, f) -> (y-f)^2, +, Y, f)
ȳ = sum(Y) / N_runs
ss_tot = sum(y -> (y-ȳ)^2, Y)
r² = 1 - ss_res / ss_tot

scatter(init_costs, optim_costs,
    xlabel = "Initial Cost", ylabel = "Final Cost", label = nothing)
plot!(x -> x*Θ[1] + Θ[2], label="r² = $(round(r², digits=3))")
savefig("guess_to_optim_scatter.svg")
savefig("guess_to_optim_scatter.png")