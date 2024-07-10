include("../ADDOPT.jl")
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA

import LinearAlgebra.mul!

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

# Geometric parameters
mask_img = Ty.(Gray.(load("scotty_bw.png")))

n_subsample = 7
mask_img_blurred = imfilter(mask_img, Kernel.gaussian(n_subsample))
mask_img_subsampled = mask_img_blurred[(n_subsample÷2):n_subsample:end, (n_subsample÷2):n_subsample:end]
mask_img_subsampled = (1.0 .- Float64.(mask_img_subsampled)) .* 0.8 .+ 0.1

mask_top = vec(mask_img_subsampled)
Nx, Ny = size(mask_img_subsampled)
Nz = 4
nsvox = Ny * Nx
nvox = nsvox * Nz
mask = vcat(mask_top, 0.1 * ones(Ty, nsvox * (Nz - 1)))
l = 1e-3 # m

@show Nx, Ny
@show nsvox
@show nvox

σ = 250e-6 / 2.355 # Spot diameter, m #250e-6
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0 # W

# Material parameters, taken at the solidus
k = 31.1e3 # W / kK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0e3 # J / kg kK 

# Tempering parameters
lnA::Ty = 38.005
n::Ty = 0.051590
E::Ty = 240.24e-3 # kJ / mol K

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 
T_AC1 = 1000.0e-3 # kK

# Environment parameters
T∞ = (20.0 + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)

dt::Ty = 40e-3 # s, aka 10ms

Tmax = T_AC1 * ones(nvox)

tempering = Tempering(lnA, n, E, dt; num=nvox)
pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M)
dynamics = PBFTempering(pbf_powerfield, tempering)

x₀ = V([T∞ * ones(nvox); log(-log(1 - 0.1)) * ones(nvox)])
uw = (ceil.(mask_top .- 0.2)) * Pₛₑₜ / sum(ceil.(mask_top .- 0.2))
uw = V(uw)

Nk = 0
x1, x2 = copy(x₀), copy(x₀)
for i in 1:1000
    transition!(dynamics, x2, x1, uw)
    if maximum(x2) ≥ T_AC1 - 100e-3
        break
    end
    Nk = i
    x1 .= x2
end
@show Nk

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, 2nvox)
b_x_eq = V(undef, 0)
A_u_eq = M(ones(1, nsvox) ./ Pₛₑₜ)
b_u_eq = V([1.0])
A_x_ineq = M([collect(1.0 * I(nvox)) zeros(nvox, nvox)])
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(nsvox)))
b_u_ineq = V(zeros(nsvox))

A_u_ineq = vcat(A_u_ineq, A_u_eq)
b_u_ineq = vcat(b_u_ineq, b_u_eq)

constraint = LinearConstraint(A_x_eq, b_x_eq, M(undef, 0, nsvox), V(undef, 0), A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
final_constraint = LinearConstraint(A_x_eq, b_x_eq, M(undef, 0, nsvox), V(undef, 0), A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
constraints = [constraint for _ in 1:(Nk-1)]
push!(constraints, final_constraint)


x̄ = V([T∞ * ones(nvox); log.(-log.(1 .- mask))])
ū = V(zeros(nsvox))

Q = diagm([zeros(nvox); 1e-5 * ones(nsvox); zeros(nvox - nsvox)])
R = zeros(nsvox, nsvox)
Q = M(Q)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)
costs = [cost for _ in 1:(Nk-1)]
final_cost = QuadraticCost(1f5 * Q, R, x̄, ū)
push!(costs, final_cost)

problem = Problem(x₀, process, costs, constraints, M)

function mul!(Y, A, B::Diagonal{T,CuVector{T}}) where {T}
    Y .= A
    Y .*= B.diag'
end

U0 = [uw for k in 1:Nk]
@time rollout!(problem, U0)
@time rollout!(problem, U0)
eval_constraints!(problem, problem.z, problem.v)
eval_penalty_multiplier!(problem, problem.v, 1e-1)
constraint_violation(problem, problem.v)
eval_lagrangian_cost(problem.v)
eval_cost(problem, problem.z)

@show eval_cost(problem)
@time al_ilqr!(problem; ctol=1e-3, μ=1e-1, ϕ=4.0, verbosity=2, ρi=1e-8, maxiters=1)
@show eval_cost(problem)

X, U = problem.z.X, problem.z.U
T_surface = [reverse(reshape(Array(x[1:nsvox]), Nx, Ny), dims=1) for x in X]
ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
P_surface = [reverse(reshape(Array(u), Nx, Ny), dims=1) for u in U]

heatmap(T_surface[end], aspect_ratio=:equal, clim=(T∞, T_AC1))
heatmap(ŷ_surface[end], aspect_ratio=:equal, clim=(-2.5, 0.75))
heatmap(reverse(log.(-log.(1 .- mask_img_subsampled)), dims=1), aspect_ratio=:equal, clim=(-2.5, 0.75))
heatmap(P_surface[end-1], aspect_ratio=:equal)
heatmap(ŷ_surface[end] .- reverse(log.(-log.(1 .- mask_img_subsampled)), dims=1), aspect_ratio=:equal, title="Error")
heatmap(abs.(ŷ_surface[end] .- reverse(log.(-log.(1 .- mask_img_subsampled)), dims=1)), aspect_ratio=:equal, title="Error")



@gif for P in P_surface
    heatmap(P, aspect_ratio=:equal, clim=(0., Pₛₑₜ / nsvox * 2))
end


@gif for T in T_surface
    heatmap(T, aspect_ratio=:equal, clim=(T∞, T_AC1))
end

@gif for y in ŷ_surface
    heatmap(y, aspect_ratio=:equal, clim=(-2.5, 0.75))
end