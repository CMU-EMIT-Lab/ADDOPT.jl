using ADDOPT
using CUDA
using LinearAlgebra
using ProgressMeter

using OptimalVoronoi
using GLMakie
using Observables

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

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
Δs = 1e-3
Δt = Δs / v             # s
ds = v * dt             # m

@show l, dt
@show r = α * dt / l^2
@assert r ≤ 1 / 6

# Create domain
Lx, Ly, Lz = 100.0e-3, 30.0e-3, 60.0e-3
Nx, Ny, Nz = round(Int, Lx / l), round(Int, Ly / l), round(Int, Lz / l)
sdf = cu(ones(Nx, Ny, Nz) * Inf)

# Create substrate
slab_th = 25.0e-3
slab_N = round(Int, slab_th / l)
sdf_box!(sdf, Nx ÷ 2, Ny ÷ 2, slab_N ÷ 2, Nx - 2, Ny - 2, slab_N)
mask = sdf .≤ 0

# Create toolpath
path = [
    20.0e-3 80.0e-3 80.0e-3+0e-3 20.0e-3+0e-3;
    15.0e-3 15.0e-3 15.0e-3+0e-3 15.0e-3+0e-3;
    slab_th slab_th slab_th+3e-3 slab_th+3e-3
]
tot_L = sum([norm(path[:, k] - path[:, k+1]) for k in 1:(size(path, 2)-1)])
Nk = floor(Int, tot_L / ds)
Nk_opt = floor(Int, tot_L / Δs)
points = path_to_points(path, ds, Nk)

# Voronoi tesselation setup, memory allocation
N_cells = 4000
sqdist = CUDA.zeros(1, N_cells)
volumes = CUDA.zeros(1, N_cells)
voronoi = sample_from_discrete_sdf(Array(sdf), N_cells)
voronoi = cu(voronoi)
Bc = CUDA.zeros(1, N_cells)
A_prealloc = CUDA.zeros(N_cells, N_cells)
e_prealloc = CUDA.zeros(N_cells)

domain1 = Int.(mask)
n_cells_1 = N_cells
domain2 = copy(domain1)
n_cells_2 = N_cells

T_cpu = zeros(Float32, Nx, Ny, Nz, Nk_opt)
voronois = zeros(Float32, 3, N_cells, Nk_opt)

P_cu = CUDA.zeros(Nx, Ny, Nz)
T_vox1_cu = CUDA.zeros(Nx, Ny, Nz)
T_vox2_cu = CUDA.zeros(Nx, Ny, Nz)

T_vox1_cu .= mask .* T∞
T_vox2_cu .= T_vox1_cu

t = 0.0
T_cpu[:, :, :, 1] .= Array(T_vox1_cu)
j = 2

dynamics_vec = []

@showprogress for i in 1:(Nk-1)
    # Update SDF and mask
    sdf_sphere!(sdf, points[1, i] / l, points[2, i] / l, points[3, i] / l, 0.003 / l)
    mask .= sdf .≤ 0
    ADDOPT.expand_bead!(T_vox1_cu, mask, Tₗ)

    p_x::Ty = round(Int, points[1, i] / l)
    p_y::Ty = round(Int, points[2, i] / l)
    p_z::Ty = round(Int, points[3, i] / l) + 0.0025 / l
    σ::Ty = 1e-3 / l

    ADDOPT.gaussian_3d!(P_cu, p_x, p_y, p_z, σ)
    P_cu .*= mask
    P_cu .*= (140.0 * 18 - ρ * cₚ * V̇ * (Tₗ - T∞)) / sum(P_cu)

    finite_diff!(T_vox2_cu, T_vox1_cu, mask, P_cu, l, dt, k, ρ, cₚ, h∞, T∞)
    T_vox1_cu .= T_vox2_cu

    t += dt
    if t > Δt
        t -= Δt
        T_cpu[:, :, :, j] .= Array(T_vox1_cu)
        j += 1

        domain2 .= Int.(mask)
        T = Ω_from_array(T_vox1_cu ./ 1e3)
        Ω = Ω_from_array(Float32.(sdf))

        voronoi = centroidal_voronoi_vox(voronoi, Ω, domain2, sqdist)
        voronoi = minimum_variance_voronoi_vox(voronoi, Ω, T, domain2, sqdist, volumes, A_prealloc, e_prealloc)
        color_voronoi!(domain2, voronoi)

        Ac, ec = mesh_fv_matrix_vector(voronoi, domain2, Ω, l, ρ, k, cₚ, h∞)
        Bc .= 0
        cell_volume_integrals!(Bc, (p, i) -> ADDOPT.gaussian_3d(p[1], p[2], p[3], p_x, p_y, p_z, σ), domain2)

        c2c = cell_to_cell_map(domain1, domain2, n_cells_1, n_cells_2)

        dynamics = WAAM_FiniteVolume(M(Array(Ac)), M(transpose(Bc)), ec .* T∞, Δt, c2c, Ty, V, M)
        domain1 .= domain2
        push!(dynamics_vec, dynamics)
    end
end


colormap = to_colormap(:thermal)
colormap[1] = RGBAf(0, 0, 0, 0)

# T_vox = Array(T_vox1_cu)
# T_vox_obv = Observable(T_vox)
# T_to_plot = @lift(view(T_vox, :, :, :, $index))
# fig, ax = volume(T_to_plot, algorithm=:mip, colorrange=(250.0, Tₗ), colormap=colormap, figure=(size=(1000, 800), fontsize=22))
# Colorbar(fig[1, 2], label="Temperature (K)", limits=(250.0, Tₗ), colormap=:thermal, flipaxis=false)

# GLMakie.record(fig, "sdf_animation.mp4", 1:4:Nk; framerate=round(Int, 1 / dt / 4)) do i
#     index[] = i
# end

# T_intp = copy(T_vox1_cu)
# cell_vals = copy(volumes)
# cell_averages!(cell_vals, volumes, domain, T)
# paint!(T_intp, domain, cell_vals)

# domain = Array(domain)
# voronoi = Array(voronoi)
# T_intp = Array(T_intp)

# fig, ax = volume(domain, colormap=colormap)
# scatter!(voronoi, label=nothing)
# display(fig)

# fig = Figure()
# l1 = LScene(fig[1, 1])
# l2 = LScene(fig[1, 2])
# volume!(l1, T_vox, colormap=colormap, colorrange=(250.0, Tₗ + 1000))
# volume!(l2, T_intp .* 1e3, colormap=colormap, colorrange=(250.0, Tₗ + 1000))
# # scatter!(l2, voronoi)
# display(fig)