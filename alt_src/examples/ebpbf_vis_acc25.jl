include("../ADDOPT.jl")
using Images
using CairoMakie
using JLD2

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

sU = [
    ("Naive", load_object("initialguess_U_5.jld2")),
    ("Optimized 1", load_object("solution_U_old_5.jld2")),
    ("Optimized 2", load_object("solution_U_new_5.jld2")),
]

Nk = length(sU[2][2])
ks = 1:4:13
n = length(ks)
m = length(sU)
Dt = 50
l = 1 * 5 / 7
clims = (1e0, 100.0)
scale = 10

fig = Figure(size=((27 * m + 20) * scale, (33 * n + 3) * scale), figure_padding=(0, 0, 10, 0), margin=0)
for (j, su) in enumerate(sU)
    str, U = su
    for (i, k) in enumerate(ks)
        ax = CairoMakie.Axis(fig[i, j], aspect=27 / 33,
            xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false,
            title=(i == 1) ? str : "", titlesize=3.6 * scale,
            ylabel=(j == 1) ? "t = $((k-1)*Dt) ms" : "", ylabelsize=2.8 * scale)
        u = reverse(reshape(max.(U[k], 1e-6) ./ l^2, 33, 27)', dims=2)
        heatmap!(ax, u; colormap=:ice, colorrange=clims, colorscale=log10)
    end
end
cb = Colorbar(fig[:, m+1], colormap=:ice, limits=clims, scale=log10,
    label="Power Density (W/mm²)", labelsize=3.6 * scale,
    ticklabelsize=2.8 * scale,
    width=6 * scale,
    ticksize=12, tickwidth=3,
    # minorticksvisible=true, minorticksize=6, minortickwidth=1.5,
    ticks=vcat(1:9, 10:10:100),
    labelpadding=0
)
save("pf_comparison.png", fig)

HV_mask = reverse(Gray.(load("scotty_target_subs5.png")), dims=1)
Nx, Ny = size(HV_mask)
Nz = 4
buffer = 2
nsvox = Ny * Nx
nvox = nsvox * Nz
Nu = (Nx - 2buffer) * (Ny - 2buffer)

l = 1e-3 * 5 / 7 # m
η = 1.0f0
σ = 1200e-6 / 1.35 # Spot diameter, m
Pₛₑₜ = 3000.0 * η # W

# Material parameters, taken at the solidus
k = 31.1e3 # W / kK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0e3 # J / kg kK 

# Environment parameters
T∞ = (20.0 + Pₛₑₜ * 0.100 * 50 / ((1 / 2) * (0.1)^2 * sqrt(k / ρ / cₚ * 5.0) * ρ * cₚ) + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)
dt::Ty = 50e-3 # s, aka 50ms

# pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; σ=σ, buffer=buffer)

x₀ = V(ones(nvox) .* T∞)
Xs = [[V(zeros(nvox)) for k in 1:Nk] for _ in sU]

for (X, su) in zip(Xs, sU)
    str, U = su
    U = [V(u) for u in U]
    rollout!([pbf_powerfield for _ in 1:Nk], Nk, x₀, X, U)
end

Xs = [[Array(x) .* 1e3 for x in X] for X in Xs]
GC.gc()


clims = (300.0, 1000.0)
fig = Figure(size=((27 * m + 20) * scale, (33 * n + 3) * scale), figure_padding=(0, 1, 10, 0), margin=0)
for (j, suX) in enumerate(zip(sU, Xs))
    su, X = suX
    str, U = su
    for (i, k) in enumerate(ks)
        ax = CairoMakie.Axis(fig[i, j], aspect=Ny / Nx,
            xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false,
            title=(i == 1) ? str : "", titlesize=3.6 * scale,
            ylabel=(j == 1) ? "t = $((k-1)*Dt) ms" : "", ylabelsize=2.8 * scale)
        u = reverse(reshape(X[k][1:nsvox], Nx, Ny)', dims=2)
        heatmap!(ax, u; colormap=:inferno, colorrange=clims)
    end
end
cb = Colorbar(fig[:, m+1], colormap=:inferno, limits=clims,
    label="Temperature (K)", labelsize=3.6 * scale,
    ticklabelsize=2.8 * scale,
    width=6 * scale,
    ticksize=12, tickwidth=3,
    # minorticksvisible=true, minorticksize=6, minortickwidth=1.5,
    labelpadding=0
)
save("temp_comparison.png", fig)