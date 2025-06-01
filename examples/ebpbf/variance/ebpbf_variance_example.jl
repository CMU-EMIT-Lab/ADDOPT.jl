using ADDOPT
using Plots
using Profile
using Images
using CUDA, LinearAlgebra
using Random, Statistics
using JLD2
using CSV, Tables

function animate_pf(t, P_surface, T_surface, name, P_max)
    anim = @animate for (t, P, T) in zip(t, P_surface, T_surface)
        title_str = @sprintf "Process at %.3f seconds" t

        h1 = heatmap(P, aspect_ratio=:equal, ticks=false, c=:ice, cbar_title="Power Density (W/mm²)", clim=(0.0, P_max))
        h2 = heatmap(T, aspect_ratio=:equal, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(T∞ * 1e3, T_melt * 1e3), title=title_str)

        plot(h1, h2; layout=(1, 2), size=((12Nx + 110) * 2, 12Ny + 110))
    end
    gif(anim, "animation_$(name).mp4", fps=(1 / (dt * 100)))
end

##### Set Up Problem #####

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

# Geometric parameters
for example_number in 1:6

buffer = 2
mask_img = Bool.(Gray.(load("examples/ebpbf/example_$(example_number).png")))
mask_reduced = mask_img[(1+buffer):(end-buffer), (1+buffer):(end-buffer)]
mask_top = vec(mask_img)
mask_reduced_top = vec(mask_reduced)
nₕ = sum(mask_top)
Nx, Ny = size(mask_img)
Nz = 4
nsvox = Ny * Nx
nvox = nsvox * Nz
mask = vcat(mask_top, zeros(Bool, nsvox * (Nz - 1)))
l = 400e-6 # m
@show nsvox
Nu = (Ny - 2buffer) * (Nx - 2buffer)

# Machine / beam parameters
σ = 250e-6 / 1.35   # Spot diameter, m
ω = 2π * 178.446e3  # 1/s
Pₛₑₜ = 3000.0e-3     # kW

# Material parameters, taken at the solidus
k = 31.1    # W / mK 
ρ = 7269.0  # kg / m^3 
cₚ = 720.0  # J / kg K, aka kJ / kg kK 

# Material critical temperatures
Tₛ = (1385.0 + 273.15) * 1e-3   # kK, solidus 
Tₗ = (1450.0 + 273.15) * 1e-3   # kK, liquidus
Tₘ = (Tₛ + Tₗ) / 2               # kK, melting 
Tboil = 3000.0e-3               # kK, boiling 

# Environment parameters
T∞ = (20.0 + 273.15) * 1e-3 # kK
T₀ = T∞                     # kK
h = 0.0                     # W / m^2 K (Vacuum)

# Timestep
dt = 1000e-6 # s, aka 1000μs

# Temperature Bounds
Tmax = Tboil * ones(nvox)
Tmax[.!mask] .= Tₛ - 25e-3
T_melt = Tₗ + 30.0e-3

# Set up dynamics for optimizer
dynamics = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; buffer=buffer)

# Initial state and input guesses
x0 = V(T∞ * ones(nvox))
ui = V(mask_reduced_top * Pₛₑₜ / nₕ)
u0 = V(zeros(Nu))

# Determine number of timesteps needed to reach melting 
Nkb = 0
Nkc = 0
x1, x2 = copy(x0), copy(x0)
for i in 1:1000
    transition!(dynamics, x2, x1, ui)
    if mean(x2[mask]) > T_melt
        break
    end
    Nkb += 1
    x1 .= x2
end
Nkc = 20
@show Nkb, Nkc
Nk = Nkb + Nkc
t = (0:(Nk-1)) .* dt

# Set up process
process = Process(Nk, [dynamics for _ in 1:Nk], V)

# Matrices and vectors for linear constraints
A_x_eq = M(undef, 0, nvox)
b_x_eq = V(undef, 0)
A_u_eq = M([collect(1.0 * I(Nu))[.!mask_reduced_top, :]; ones(1, Nu) ./ Pₛₑₜ])
b_u_eq = V([zeros(Nu - nₕ); 1.0])
A_x_ineq = M(collect(1.0 * I(nvox)))
b_x_ineq = V(Tmax)
A_u_ineq = M(collect(-1.0 * I(Nu)))
b_u_ineq = V(zeros(Nu))

# Set up constraint objects
build_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
cooling_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_ineq, b_u_ineq, A_x_ineq, b_x_ineq, M(undef, 0, Nu), V(undef, 0))
constraints = vcat([build_constraint for _ in 1:Nkb], [cooling_constraint for _ in 1:Nkc])

# Set up costs
x̄ = V(zeros(nvox))
ū = V(zeros(Nu))
Q = M((diagm(mask) - (1 / nₕ) * (mask * mask'))' * (diagm(mask) - (1 / nₕ) * (mask * mask')) / nₕ)
R = M(zeros(Nu, Nu))
cost = QuadraticCost(Q, R, x̄, ū)
costs = [cost for _ in 1:Nk]

# Set up problem for optimizer
problem = Problem(x0, process, costs, constraints, M)

U0 = vcat([ui for k in 1:Nkb], [u0 for k in 1:Nkc])
rollout!(problem, U0)
U_unopt = [Array(u) for u in problem.z.U]
X_unopt = [Array(x) for x in problem.z.X]

#### Visualize Initial Guess #####
T_surface_unopt = [reshape(reverse(x[1:nsvox], dims=1) .* 1e3, Nx, Ny) for x in X_unopt]
P_surface_unopt = [reshape(reverse(u, dims=1) .* 1e3 / (l * 1e3)^2, Nx - 2buffer, Ny - 2buffer) for u in U_unopt]
P_max = round(maximum([maximum(P) for P in P_surface_unopt]), sigdigits=1)
# animate_pf(t, P_surface_unopt, T_surface_unopt, "unoptimized_$(example_number)", P_max)

nvox_fine = 2Nx * 2Ny * Nz÷2
Nu_fine = 2(Nx-2buffer) * 2(Ny-2buffer)
steps_per_spot = 50
tmin = 1e-6 / 10
mask_fine = refine_grid(reshape(mask_reduced_top, Nx-2buffer, Ny-2buffer), 2)
xyz = get_coordinates(2(Nx-2buffer), 2(Ny-2buffer), 1, l/2)
px = xyz[1, :]
py = xyz[2, :]

U_unopt_build = problem.z.U[1:Nkb]
U_unopt_build_fine = [cu(vec(refine_grid(reshape(Array(u), Nx-2buffer, Ny-2buffer), 2))) ./ 2^2 for u in U_unopt_build]

seq, dwell = powerfield_to_sequence_with_traverse(dt, 1 / ω, tmin, U_unopt_build_fine, px, py, l/2, Pₛₑₜ, σ; steps_per_spot=steps_per_spot)
points = [(xyz[:, seq][1], xyz[:, seq][2]) for seq in seq]
CSV.write("scan_strat_unoptimized_$(example_number).csv", Tables.table(vcat([[round(x, digits=9); round(y, digits=9); round(dwell, digits=9)]' for (x, y) in points]...); header=["X", "Y", "Δt"]))


##### Run Optimization #####
@show eval_cost(problem)
@time al_ddp!(problem; ctol=1e-6, μ=0.1, ϕ=3.0, verbosity=1, ρi=1e-10, tol=1e-6, gtol=Pₛₑₜ / Nu / 100)
@show eval_cost(problem)
U_opt = [Array(u) for u in problem.z.U]
X_opt = [Array(x) for x in problem.z.X]

#### Visualize Optimal Solution #####
T_surface_opt = [reshape(reverse(x[1:nsvox], dims=1) .* 1e3, Nx, Ny) for x in X_opt]
P_surface_opt = [reshape(reverse(u, dims=1) .* 1e3 / (l * 1e3)^2, Nx - 2buffer, Ny - 2buffer) for u in U_opt]
P_max = round(maximum([maximum(P) for P in P_surface_opt]), sigdigits=1)
# animate_pf(t, P_surface_opt, T_surface_opt, "optimized_$(example_number)", P_max)

##### Calculate Statistics #####
T0m = [mean(x[mask]) * 1e3 for x in X_unopt]
T0σ = [std(x[mask]) * 1e3 for x in X_unopt]
Tom = [mean(x[mask]) * 1e3 for x in X_opt]
Toσ = [std(x[mask]) * 1e3 for x in X_opt]

##### Visualize Standard Deviation #####
# std_comp = plot(xlabel="Time (ms)", ylabel="Standard Deviation of Temperature (K)", tickfontsize=14, labelfontsize=16, legendfontsize=14, size=(800, 800), widen=true, handlelength=8, grid=false)
# plot!(std_comp, x_foreground_color_axis=:black, y_foreground_color_axis=:black)
# plot!(std_comp, t .* 1e3, T0σ, linewidth=3, thickness_scaling=1, label="Uniform Power, Ideal", c="lightblue")
# plot!(std_comp, t .* 1e3, Toσ, linewidth=3, thickness_scaling=1, label="Optimized Power, Ideal", c="indianred")

##### Visualize Cumulative Variance #####
# var_comp_int = 

U_opt_build = problem.z.U[1:Nkb]
U_opt_build_fine = [cu(vec(refine_grid(reshape(Array(u), Nx-2buffer, Ny-2buffer), 2))) ./ 2^2 for u in U_opt_build]

problem = 0 
process = 0
dynamics = 0
GC.gc()
CUDA.reclaim()

seq, dwell = powerfield_to_sequence_with_traverse(dt, 1 / ω, tmin, U_opt_build_fine, px, py, l/2, Pₛₑₜ, σ; steps_per_spot=steps_per_spot)
points = [(xyz[:, seq][1], xyz[:, seq][2]) for seq in seq]
CSV.write("scan_strat_optimized_$(example_number).csv", Tables.table(vcat([[round(x, digits=9); round(y, digits=9); round(dwell, digits=9)]' for (x, y) in points]...); header=["X", "Y", "Δt"]))

# u0_fine = cu(zeros(Nu_fine))
# f = round(Int, dt / tmin)
# U_approx = [U_approx[i, :] for i in 1:size(U_approx, 1)]
# U_approx = vcat(U_approx, [u0_fine for k in 1:(Nkc*f)])
# X_approx = [CUDA.zeros(nvox_fine) for i in 1:(Nk*f)]

# dynamics_approx = PBFPowerField(2Nx, 2Ny, Nz÷2, l/2, k, ρ, cₚ, T∞, T₀, h, tmin, Ty, V, M; buffer=2buffer)

# rollout!([dynamics_approx for k in 1:(Nk*f)], Nk * f, cu(ones(nvox_fine).*T∞), X_approx, U_approx)
# U_approx = [Array(u) for u in view(U_approx, 1:100:(Nk*f))]
# X_approx = [Array(x) for x in view(X_approx, 1:100:(Nk*f))]
# t_approx = (1:100:(Nk*f)) .* tmin
# T_surface_approx = [reshape(reverse(x[1:(2Nx*2Ny)], dims=1) .* 1e3, 2Nx, 2Ny) for x in X_approx]
# P_surface_approx = [reshape(reverse(u, dims=1) .* 1e3 / (l/2 * 1e3)^2, 2(Nx - 2buffer), 2(Ny - 2buffer)) for u in U_approx]
# P_max = round(maximum([maximum(P) for P in P_surface_approx]), sigdigits=1)

# animate_pf(t_approx, P_surface_approx, T_surface_approx, "approximated_$(example_number)", P_max)

seq = shuffle(findall(>(0), vec(mask_fine)))
points = [(xyz[:, seq][1], xyz[:, seq][2]) for seq in seq]
CSV.write("scan_strat_random_$(example_number).csv", Tables.table(vcat([[round(x, digits=9); round(y, digits=9); round(50e-6, digits=9)]' for (x, y) in points]...); header=["X", "Y", "Δt"]))

end