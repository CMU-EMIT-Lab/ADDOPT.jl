using ADDOPT
using CUDA
using LinearAlgebra
using ProgressMeter

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
l = 0.25e-3
dt = l^2 * 0.1 / α
# dt = 0.01                # s
ds = v * dt             # m

@show l, dt
@show r = α * dt / l^2
@assert r ≤ 1 / 2

slab_th = 25.0e-3
Lx, Ly, Lz = 100.0e-3, 30.0e-3, 60.0e-3
slab_N = round(Int, slab_th / l)
Nx, Ny, Nz = round(Int, Lx / l), round(Int, Ly / l), round(Int, Lz / l)

sdf = cu(ones(Nx, Ny, Nz) * Inf)
sdf_box!(sdf, Nx ÷ 2, Ny ÷ 2, slab_N ÷ 2, Nx - 2, Ny - 2, slab_N)

mask = sdf .≤ 0

path = [
    20.0e-3 80.0e-3 80.0e-3+0e-3 20.0e-3+0e-3;
    15.0e-3 15.0e-3 15.0e-3+0e-3 15.0e-3+0e-3;
    slab_th slab_th slab_th+3e-3 slab_th+3e-3
]
tot_L = sum([norm(path[:, k] - path[:, k+1]) for k in 1:(size(path, 2)-1)])
Nk = floor(Int, tot_L / ds)
points = path_to_points(path, ds, Nk)

P_cu = CUDA.zeros(Nx, Ny, Nz)
T_vox1_cu = CUDA.zeros(Nx, Ny, Nz)
T_vox2_cu = CUDA.zeros(Nx, Ny, Nz)

T_vox1_cu .= mask .* T∞
T_vox2_cu .= T_vox1_cu

@showprogress for i in 1:(Nk-1)
    # Update SDF and mask
    sdf_sphere!(sdf, points[1, i] / l, points[2, i] / l, points[3, i] / l, 0.003 / l)
    mask .= sdf .≤ 0
    ADDOPT.expand_bead!(T_vox1_cu, mask, Tₗ)

    ADDOPT.gaussian_3d!(P_cu, round(Int, points[1, i] / l), round(Int, points[2, i] / l), round(Int, points[3, i] / l) + 0.0025 / l, 1e-3 / l)
    P_cu .*= mask
    P_cu .*= (140.0 * 18 - ρ * cₚ * V̇ * (Tₗ - T∞)) / sum(P_cu)

    finite_diff!(T_vox2_cu, T_vox1_cu, mask, P_cu, l, dt, k, ρ, cₚ, h∞, T∞)
    T_vox1_cu .= T_vox2_cu
end

T_vox = Array(T_vox1_cu)

# T_vox_obv = Observable(T_vox)
# T_to_plot = @lift(view(T_vox, :, :, :, $index))

colormap = to_colormap(:thermal)
colormap[1] = RGBAf(0, 0, 0, 0)
# fig, ax = volume(T_to_plot, algorithm=:mip, colorrange=(250.0, Tₗ), colormap=colormap, figure=(size=(1000, 800), fontsize=22))
# Colorbar(fig[1, 2], label="Temperature (K)", limits=(250.0, Tₗ), colormap=:thermal, flipaxis=false)

# GLMakie.record(fig, "sdf_animation.mp4", 1:4:Nk; framerate=round(Int, 1 / dt / 4)) do i
#     index[] = i
# end

N_cells = 4000
points = sample_from_discrete_sdf(Array(sdf), N_cells)

# scatter(Array(points))
# volume!(Array(sdf), algorithm=:iso, isovalue=0, isorange=0.1, alpha=0.1)

# Voronoi stuff

Ω = Ω_from_array(Float32.(sdf))
T = Ω_from_array(T_vox1_cu ./ 5e2)

domain = Int.(mask)
sqdist = CUDA.zeros(1, N_cells)
volumes = CUDA.zeros(1, N_cells)
points = cu(points)
A = CUDA.zeros(N_cells, N_cells)
e = CUDA.zeros(N_cells)

points = centroidal_voronoi_vox(points, Ω, domain, sqdist);
voronoi = minimum_variance_voronoi_vox(points, Ω, T, domain, sqdist, volumes, A, e);

color_voronoi!(domain, voronoi)

Ac, ec = mesh_fv_matrix_vector(points, domain, Ω, l, ρ, k, cₚ, h∞)

T_intp = copy(T_vox1_cu)
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
volume!(l1, T_vox, colormap=colormap, colorrange=(250.0, Tₗ))
volume!(l2, T_intp .* 5e2, colormap=colormap, colorrange=(250.0, Tₗ))
# scatter!(l2, voronoi)
display(fig)