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
mask_img = Bool.(Gray.(load("trial_old.png")))
mask_top = vec(mask_img)
nₕ = sum(mask_top)
Nx, Ny = size(mask_img)
Nz = 4
nsvox = Ny * Nx
nvox = nsvox * Nz
mask = vcat(mask_top, zeros(Bool, nsvox * (Nz - 1)))
l = 400e-6 # m
@show nsvox

Nk = 80

σ = 250e-6 / 2.355 # Spot diameter, m #250e-6
ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0e-3 # kW

# Material parameters, taken at the solidus
k = 31.1 # W / mK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0 # J / kg K, aka kJ / kg kK 

Tₛ = (1385.0 + 273.15) * 1e-3 # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3 # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2 # kK, melting 
Tboil = 3000.0e-3 # kK, boiling 

# Environment parameters
T∞ = (20.0 + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)

dt = 1000e-6 # s, aka 1000μs

Tmax = Tboil * ones(nvox)
Tmax[.!mask] .= Tₛ - 25e-3

dynamics = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M)

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, nvox)
b_x_eq = V(undef, 0)
A_u_eq = M([collect(1.0 * I(nsvox))[.!mask_top, :]; ones(1, nsvox)])
b_u_eq = V([zeros(nsvox - nₕ); Pₛₑₜ])
A_x_ineq = M(collect(1.0 * I(nvox)))
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(nsvox)))
b_u_ineq = V(zeros(nsvox))

constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

x₀ = T∞ * V(ones(nvox))
x̄ = V(zeros(nvox))
ū = V(zeros(nsvox))

Q = (diagm(mask) - (1 / nₕ) * (mask * mask'))' * (diagm(mask) - (1 / nₕ) * (mask * mask')) / nₕ
# Q = (I - (1 / nvox) * (ones(nvox) * ones(nvox)'))' * (I - (1 / nvox) * (ones(nvox) * ones(nvox)')) / nvox
R = zeros(nsvox, nsvox)
Q = M(Q)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)


problem = Problem(x₀, process, [cost for _ in 1:Nk], [constraint for _ in 1:Nk], M)
uw = mask_top * Pₛₑₜ / nₕ
uw = V(uw)

function mul!(Y, A, B::Diagonal{T, CuVector{T}}) where T
    Y .= A
    Y .*= B.diag'
end

U0 = [uw for k in 1:Nk]
@time rollout!(problem, U0)
@time rollout!(problem, U0)

@time al_ilqr!(problem, ϕ=2.0, verbosity=1)

for k in 1:Nk
    problem.v.λ[k] .= 0
    problem.v̄.λ[k] .= 0
end

@time al_ilqr!(problem, ϕ=2.0, verbosity=1)

X, U = problem.z.X, problem.z.U
T_surface = [reshape(Array(x[1:nsvox]), Nx, Ny) for x in X]
P_surface = [reshape(Array(u), Nx, Ny) for u in U]
heatmap(T_surface[end], aspect_ratio=:equal)
heatmap(P_surface[end-2], aspect_ratio=:equal)