using ADDOPT
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using CSV, Tables
using Printf
using JLD2
using ColorSchemes

HVmin = 135.0
HVmax = 404.0
# HVmin = 140.0
# HVmax = 400.0
y_init = (HVmax - 365) / (HVmax - HVmin)
# y_init = (HVmax - 380) / (HVmax - HVmin)

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

tempering = Tempering(lnA, n, E, dt; num=nvox)
pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; σ=σ, buffer=buffer)
dynamics = PBFTempering(pbf_powerfield, tempering)

x₀ = V([T∞ * ones(nvox); log(-log(1 - y_init)) * ones(nvox)])

U_opt = load_object("examples/ebpbf/solution_U_new_$(n_subsample).jld2")
# U_opt = load_object("solution_U_1_$(n_subsample).jld2")
Nk = length(U_opt)
dynamics_v = [dynamics for _ in 1:Nk]

X = [V(zeros(2nvox)) for _ in 1:Nk]
U = [V(u) for u in U_opt]

rollout!(dynamics_v, Nk, x₀, X, U)
t = (0:(Nk-1)) .* dt

### VISUALIZE AFTER OPTIMIZATION ###
T_surface = [reverse(reshape(Array(x[1:nsvox] .* 1e3), Nx, Ny), dims=1) for x in X]
ŷ_surface = [reverse(reshape(Array(x[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1) for x in X]
P_surface = [reverse(reshape(Array((pbf_powerfield.B*u)[1:nsvox] .* (l^3 * ρ * cₚ) ./ (l * 1e3)^2), Nx, Ny), dims=1) for u in U]
Pdmax = round(maximum([maximum(P) for P in P_surface]), sigdigits=1)

y_surface = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface]
H_end = y_surface[end] * (HVmin - HVmax) .+ HVmax

hm = heatmap(H_end, aspect_ratio=:equal, clim=(140, 420), size=(550, 600), fontsize=16, tickfontsize=14)
savefig("hardness_fig.png")

cg = cgrad(:default, [140, 420])

T1 = [T[end-3, Ny÷2] for T in T_surface]
T2 = [T[end-6, Ny÷2] for T in T_surface]
T3 = [T[end-14, Ny÷2] for T in T_surface]

P1 = [P[end-3, Ny÷2] for P in P_surface]
P2 = [P[end-6, Ny÷2] for P in P_surface]
P3 = [P[end-14, Ny÷2] for P in P_surface]

c1 = cg[Int((H_end[end-3, Ny÷2] - 140) * 256 ÷ (420 - 140))]
c2 = cg[Int((H_end[end-6, Ny÷2] - 140) * 256 ÷ (420 - 140))]
c3 = cg[Int((H_end[end-14, Ny÷2] - 140) * 256 ÷ (420 - 140))]

default(fontfamily="Times_New_Roman",
    linewidth=2, framestyle=:box, grid=false)

plot(xlabel="Time (s)", ylabel="Temperature (K)", size=(800, 600),
    labelfontsize=16, tickfontsize=14, legendfontsize=14)
plot!(t, T1, label="Border (Med. Soft)", color=c1)
plot!(t, T2, label="Interior (Hard)", color=c2)
plot!(t, T3, label="Interior (Very Soft)", color=c3)
savefig("temperature_plot.png")


plot(xlabel="Time (s)", ylabel="Power Density (W/mm²)", size=(800, 600),
    labelfontsize=16, tickfontsize=14, legendfontsize=14)
plot!(t, P1, label="Border (Med. Soft)", color=c1)
plot!(t, P2, label="Interior (Hard)", color=c2)
plot!(t, P3, label="Interior (Very Soft)", color=c3)
savefig("power_plot.png")
