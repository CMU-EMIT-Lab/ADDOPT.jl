include("../ADDOPT.jl")
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using Printf

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

l = 1e-3 * n_subsample / 7 # m

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

dt::Ty = 50e-3 # s, aka 50ms

Tmax = T_AC1 * ones(nvox)

tempering = Tempering(lnA, n, E, dt; num=nvox)
pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M)
dynamics = PBFTempering(pbf_powerfield, tempering)

x₀ = V([T∞ * ones(nvox); log(-log(1 - 0.1)) * ones(nvox)])
uw = (mask_top .+ 0.2) * Pₛₑₜ / sum((mask_top .+ 0.2))
uw = V(uw)
u0 = zeros(nsvox)
u0 = V(u0)

Nk = 0
Nkc = 0
x1, x2 = copy(x₀), copy(x₀)
for i in 1:1000
    transition!(dynamics, x2, x1, uw)
    if maximum(x2) ≥ T_AC1 - 100e-3
        break
    end
    Nk = i
    x1 .= x2
end
for i in 1:1000
    transition!(dynamics, x2, x1, u0)
    if maximum(x2) ≤ 450e-3
        break
    end
    Nkc = i
    x1 .= x2
end
@show Nk = Nk + Nkc

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, 2nvox)
b_x_eq = V(undef, 0)
A_u_eq = M(ones(1, nsvox) ./ Pₛₑₜ)
b_u_eq = V([1.0])
A_x_ineq = M([collect(1.0 * I(nvox)) zeros(nvox, nvox)])
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(nsvox)))
b_u_ineq = V(zeros(nsvox))

build_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
cooling_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_ineq, b_u_ineq, A_x_ineq, b_x_ineq, M(undef, 0, nsvox), V(undef, 0))
constraints = vcat([build_constraint for _ in 1:(Nk-Nkc)], [cooling_constraint for _ in 1:Nkc])


x̄ = V([T∞ * ones(nvox); log.(-log.(1 .- mask))])
ū = V(zeros(nsvox))

Q = diagm([zeros(nvox); 1e-2 / Nk / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
Qf = diagm([zeros(nvox); 1e3 / nsvox * ones(nsvox); 0 * ones(nvox - nsvox)])
R = diagm(0.0 * ones(nsvox))
Q = M(Q)
Qf = M(Qf)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)
costs = [cost for _ in 1:(Nk-1)]
final_cost = QuadraticCost(Qf, R, x̄, ū)
push!(costs, final_cost)

problem = Problem(x₀, process, costs, constraints, M)

function mul!(Y, A, B::Diagonal{T,CuVector{T}}) where {T}
    Y .= A
    Y .*= B.diag'
end

U0 = vcat([uw for k in 1:(Nk-Nkc)], [u0 for k in 1:Nkc])
@time rollout!(problem, U0)
@time rollout!(problem, U0)
eval_constraints!(problem, problem.z, problem.v)
eval_penalty_multiplier!(problem, problem.v, 1e-1)
@show constraint_violation(problem, problem.v)
eval_lagrangian_cost(problem.v)

### VISUALIZE BEFORE OPTIMIZATION ###
X, U = problem.z.X, problem.z.U
t = (0:(Nk-1)) .* dt
T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X]
ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
P_surface = [reverse(reshape(Array(u ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U]
Pdmax = round(maximum([maximum(P) for P in P_surface]), sigdigits=1)

y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]
heatmap(abs.(y_surface[end] .- reverse(mask_img_subsampled, dims=1)), aspect_ratio=:equal, title="Error Magnitude")

anim = @animate for (t, P, T, y) in zip(t, P_surface, T_surface, y_surface)
    title_str = @sprintf "Process at %.3f seconds" t

    h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, Pdmax))
    h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_AC1 * 1e3), title=title_str)
    h3 = heatmap(y, aspect_ratio=:equal, ticks=false, c=:acton, cbar_title="Fraction Tempered", clim=(0, 1))

    plot(h1, h2, h3; layout=(1, 3), size=((12Nx) * 3, 12Ny + 110))
end
gif(anim, "unoptimized_$(n_subsample).mp4", fps=(1 / dt))


### OPTIMIZE ###
@show eval_cost(problem)
@time al_ilqr!(problem; ctol=1e-4, μ=0.1, ϕ=3.0, verbosity=2, ρi=1e-10, tol=1e-4, gtol=Pₛₑₜ / nsvox / 100)
# @profview al_ilqr!(problem; ctol=1e-4, μ=0.1, ϕ=3.0, verbosity=2, ρi=1e-10, tol=1e-4, gtol=Pₛₑₜ / nsvox / 100)
@show eval_cost(problem)


### VISUALIZE AFTER OPTIMIZATION ###
X, U = problem.z.X, problem.z.U
t = (0:(Nk-1)) .* dt
T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X]
ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
P_surface = [reverse(reshape(Array(u ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U]
Pdmax = round(maximum([maximum(P) for P in P_surface]), sigdigits=1)

y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]

# heatmap(T_surface[end], aspect_ratio=:equal, clim=(T∞, T_AC1))
# heatmap(ŷ_surface[end], aspect_ratio=:equal, clim=(-2.5, 0.75))
# heatmap(reverse(log.(-log.(1 .- mask_img_subsampled)), dims=1), aspect_ratio=:equal, clim=(-2.5, 0.75))
# heatmap(P_surface[end-1], aspect_ratio=:equal)
# heatmap(ŷ_surface[end] .- reverse(log.(-log.(1 .- mask_img_subsampled)), dims=1), aspect_ratio=:equal, title="Error")
heatmap(abs.(y_surface[end] .- reverse(mask_img_subsampled, dims=1)), aspect_ratio=:equal, title="Error Magnitude")


anim = @animate for (t, P, T, y) in zip(t, P_surface, T_surface, y_surface)
    title_str = @sprintf "Process at %.3f seconds" t

    h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, Pdmax))
    h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_AC1 * 1e3), title=title_str)
    h3 = heatmap(y, aspect_ratio=:equal, ticks=false, c=:acton, cbar_title="Fraction Tempered", clim=(0, 1))

    plot(h1, h2, h3; layout=(1, 3), size=((12Nx) * 3, 12Ny + 110))
end
gif(anim, "optimized_$(n_subsample).mp4", fps=(1 / dt))

# save_object("solution_subsampled_$(n_subsample).jld2", P_surface)