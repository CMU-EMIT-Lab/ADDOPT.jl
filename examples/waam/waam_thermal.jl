using ADDOPT
using CUDA

using OptimalVoronoi
using GLMakie
using Observables

# Material properties
k = 34.0                # W / m⋅K
ρ = 7826.0              # kg / m³
cₚ = 502.416            # J / kg⋅K 
α = k / ρ / cₚ          # m² / s
Tₗ = 1784.0             # K, liquidus
Tmax = 3000.0           # K, boiling

# Environmental properties
h∞ = 10.0               # W / m²⋅K
hₐᵣ = 500.0             # W / m²⋅K
T∞ = 295.0              # K, ambient
η = 0.8

# Wire and print properties
d = 0.001143            # m, aka 0.045in
wfs = 67.7e-3           # m / s
v = 5.0e-3              # m / s
V̇ = π / 4 * d^2 * wfs   # m³ / s
A = V̇ / v               # m²

# Discretization (temporal and spatial)
dt = 0.01                # s
ds = v * dt             # m
Fo = α * dt / ds^2
l = 1e-3

@show r = α * dt / l^2
@assert r ≤ 1 / 2

sdf = cu(ones(80, 80, 100) * Inf)
sdf_box!(sdf, 40, 40, 20, 78, 78, 38)

mask = sdf .≤ 0

Nk = 1800
path = [
    40.0e-3 65.0e-3 40.0e-3 40.0e-3 65.0e-3;
    40.0e-3 40.0e-3 65.0e-3 65.0e-3 40.0e-3;
    39.0e-3 39.0e-3 39.0e-3 41.5e-3 41.5e-3
]
points = path_to_points(path, ds, Nk)

P_cu = CUDA.zeros(80, 80, 100)
T_vox_cu = CUDA.zeros(80, 80, 100, Nk); #[mask .* T∞ for k in 1:Nk]

T_vox_cu .= mask .* T∞;

T_vox_cud(i) = view(T_vox_cu, :, :, :, i)

for i in 1:(Nk-1)
    # Update SDF and mask
    sdf_sphere!(sdf, points[1, i] / l, points[2, i] / l, points[3, i] / l, 0.003 / l)
    mask .= sdf .≤ 0
    ADDOPT.expand_bead!(T_vox_cud(i), mask, Tₗ)

    ADDOPT.gaussian_3d!(P_cu, round(Int, points[1, i] / l), round(Int, points[2, i] / l), round(Int, points[3, i] / l) + 0.0025 / l, 1e-3 / l)
    P_cu .*= mask
    P_cu .*= (140.0 * 18 - ρ * cₚ * V̇ * (Tₗ - T∞)) / sum(P_cu)

    finite_diff!(T_vox_cud(i + 1), T_vox_cud(i), mask, P_cu, l, dt, k, ρ, cₚ, h∞, T∞)
end

T_vox = Array(T_vox_cu)

index = Observable(1)
T_to_plot = @lift(view(T_vox, :, :, :, $index))

colormap = to_colormap(:thermal)
colormap[1] = RGBAf(0, 0, 0, 0)
fig, ax = volume(T_to_plot, algorithm=:mip, colorrange=(250.0, Tₗ), colormap=colormap, figure=(size=(1000, 800), fontsize=22))
Colorbar(fig[1, 2], label="Temperature (K)", limits=(250.0, Tₗ), colormap=:thermal, flipaxis=false)

GLMakie.record(fig, "sdf_animation.mp4", 1:4:Nk; framerate=round(Int, 1 / dt / 4)) do i
    index[] = i
end

N_cells = 1000
points = sample_from_discrete_sdf(Array(sdf), N_cells)

# scatter(Array(points))
volume!(Array(sdf), algorithm=:iso, isovalue=0, isorange=0.1, alpha=0.1)

## Voronoi stuff

Ω = Ω_from_array(Float32.(sdf))
T = Ω_from_array(T_vox_cu[:, :, :, end] ./ 1e2)

domain = Int.(mask)
sqdist = CUDA.zeros(1, N_cells)
volumes = CUDA.zeros(1, N_cells)
points = cu(points)
A = CUDA.zeros(N_cells, N_cells)
e = CUDA.zeros(N_cells)

# voronoi = centroidal_voronoi_vox(points, Ω, domain, sqdist);
voronoi = minimum_variance_voronoi_vox(points, Ω, T, domain, sqdist, volumes, A, e);

color_voronoi!(domain, voronoi)

T_intp = copy(T_vox_cu[:, :, :, end])
cell_vals = copy(volumes)
cell_averages!(cell_vals, volumes, domain, T)
paint!(T_intp, domain, cell_vals)

domain = Array(domain)
voronoi = Array(voronoi)
T_intp = Array(T_intp)

fig, ax = volume(domain, colormap=colormap)
scatter!(voronoi, label=nothing)
display(fig)

fig = Figure()
l1 = LScene(fig[1, 1])
l2 = LScene(fig[1, 2])
volume!(l1, T_vox[:, :, :, end], colormap=colormap, colorrange=(250.0, Tₗ))
volume!(l2, T_intp .* 1e2, colormap=colormap, colorrange=(250.0, Tₗ))
scatter!(l2, voronoi)
display(fig)