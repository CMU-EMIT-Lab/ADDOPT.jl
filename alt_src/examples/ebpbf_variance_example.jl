include("../ADDOPT.jl")
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using Random
using JLD2

import LinearAlgebra.mul!
function mul!(Y, A, B::Diagonal{T,CuVector{T}}) where {T}
    Y .= A
    Y .*= B.diag'
end

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

buffer = 3
# Geometric parameters
mask_img = Bool.(Gray.(load("scotty_mask.png")))
mask_red = mask_img[(1+buffer):(end-buffer), (1+buffer):(end-buffer)]
mask_top = vec(mask_img)
mask_red_top = vec(mask_red)
nₕ = sum(mask_top)
Nx, Ny = size(mask_img)
Nz = 4
nsvox = Ny * Nx
nvox = nsvox * Nz
mask = vcat(mask_top, zeros(Bool, nsvox * (Nz - 1)))
l = 200e-6 # m
@show nsvox
Nu = (Ny - 2buffer) * (Nx - 2buffer)

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

dt = 3000e-6 # s, aka 1000μs

Tmax = Tboil * ones(nvox)
Tmax[.!mask] .= Tₛ - 25e-3

dynamics = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; buffer=buffer)

process = Process(Nk, [dynamics for _ in 1:Nk], V)

A_x_eq = M(undef, 0, nvox)
b_x_eq = V(undef, 0)
A_u_eq = M([collect(1.0 * I(Nu))[.!mask_red_top, :]; ones(1, Nu)])
b_u_eq = V([zeros(Nu - nₕ); Pₛₑₜ])
A_x_ineq = M(collect(1.0 * I(nvox)))
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(Nu)))
b_u_ineq = V(zeros(Nu))

constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

x₀ = T∞ * V(ones(nvox))
x̄ = V(zeros(nvox))
ū = V(zeros(Nu))

Q = (diagm(mask) - (1 / nₕ) * (mask * mask'))' * (diagm(mask) - (1 / nₕ) * (mask * mask')) / nₕ
# Q = (I - (1 / nvox) * (ones(nvox) * ones(nvox)'))' * (I - (1 / nvox) * (ones(nvox) * ones(nvox)')) / nvox
R = zeros(Nu, Nu)
Q = M(Q)
R = M(R)
cost = QuadraticCost(Q, R, x̄, ū)


problem = Problem(x₀, process, [cost for _ in 1:Nk], [constraint for _ in 1:Nk], M)
uw = mask_red_top * Pₛₑₜ / nₕ
uw = V(uw)

U0 = [uw for k in 1:Nk]
@time rollout!(problem, U0)
@time rollout!(problem, U0)

@time al_ilqr!(problem, ϕ=2.0, verbosity=2, ctol=1e-4)

X, U = problem.z.X, problem.z.U
T_surface_pf = [reshape(reverse(Array(x[1:nsvox]), dims=1), Nx, Ny) for x in X]
P_surface_pf = [reshape(reverse(Array(u), dims=1), Nx - 2buffer, Ny - 2buffer) for u in U]

function plot_history(t, T_arr, stratname; nsubdiv=4)
    plots = []
    Nk = length(T_arr)
    for k in 1:nsubdiv
        T = T_arr[Nk*k÷nsubdiv]
        p = heatmap(T, aspect_ratio=:equal, clim=(0.5, Tₗ + 0.1), colorbar=nothing, ticks=nothing,
            title="t=$(round(t[Nk*k÷nsubdiv], digits=6))s", titlefontsize=18)
        push!(plots, p)
    end
    p = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=(T∞, Tₗ + 0.1),
        xlims=(1, 1.1), label="", colorbar_title="Temperature (kK)",
        framestyle=:none, colorbar_titlefontsize=16, ytickfontsize=14)
    push!(plots, p)
    lay = @layout [grid(1, nsubdiv) a{0.035w}]
    p = plot(plots..., layout=lay, size=(8 * (Ny * nsubdiv) + 220, 8 * Nx + 30))
    savefig("thermal_history_$(stratname).png")
end


problem = dynamics = process = 0
GC.gc()

dt_n = 60e-6 # s
dynamics_n = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt_n, Ty, V, M; buffer=buffer)

idxs = findall(!iszero, mask_red_top)
U_raster = [zeros(Nu) for k in 1:nₕ]
for k in 1:nₕ
    U_raster[k][idxs[k]] = Pₛₑₜ
end
U_raster = [V(u) for u in U_raster]
X_raster = [V(zeros(nvox)) for k in 1:nₕ]

rollout!([dynamics_n for k in 1:nₕ], nₕ, V(x₀), X_raster, U_raster)
T_surface_raster = [reshape(reverse(Array(x[1:nsvox]), dims=1), Nx, Ny) for x in X_raster]
P_surface_raster = [reshape(reverse(Array(u), dims=1), Nx - 2buffer, Ny - 2buffer) for u in U_raster]

U_spotmelt = shuffle(U_raster)
X_spotmelt = [V(zeros(nvox)) for k in 1:nₕ]
rollout!([dynamics_n for k in 1:nₕ], nₕ, V(x₀), X_spotmelt, U_spotmelt)
T_surface_spotmelt = [reshape(reverse(Array(x[1:nsvox]), dims=1), Nx, Ny) for x in X_spotmelt]
P_surface_spotmelt = [reshape(reverse(Array(u), dims=1), Nx - 2buffer, Ny - 2buffer) for u in U_spotmelt]


T_surface_pf = load_object("T_pf.jld2")
T_surface_raster = load_object("T_raster.jld2")
T_surface_spotmelt = load_object("T_spotmelt.jld2")

xvox, yvox = Nx÷2, Ny÷2
T_samp_pf = [T[xvox,yvox] for T in T_surface_pf]
T_samp_raster = [T[xvox,yvox] for T in T_surface_raster]
T_samp_spotmelt = [T[xvox,yvox] for T in T_surface_spotmelt]
t_pf = dt * (1:Nk) .- dt
t_raster = dt_n * (1:nₕ) .- dt_n
t_spotmelt = dt_n * (1:nₕ) .- dt_n

plot_history(t_pf, T_surface_pf, "powerfield")
plot_history(t_raster, T_surface_raster, "raster")
plot_history(t_spotmelt, T_surface_spotmelt, "spotmelt")

plot(xlabel="Time (s)", ylabel="Temperature (kK)")
plot!(t_pf, T_samp_pf, label="Power Field")
plot!(t_raster, T_samp_raster, label="Raster")
plot!(t_spotmelt, T_samp_spotmelt, label="Spot Melt")
savefig("point_comparison.png")

save_object("T_pf.jld2", T_surface_pf)
save_object("T_raster.jld2", T_surface_raster)
save_object("T_spotmelt.jld2", T_surface_spotmelt)

@gif for T in T_surface_spotmelt
    heatmap(T, aspect_ratio=:equal, clim=(.500, 2.100))
end
