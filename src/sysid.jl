using DataFrames
using CSV: File, read
using Plots
using Statistics
using StringEncodings
using Interpolations
using JuMP
using ForwardDiff
using Ipopt
import HSL_jll

# import MathOptInterface
using .animation_measured3d4_optimized# const MOI = MathOptInterface

struct SysidProblem <: MOI.AbstractNLPEvaluator
    X::Vector{Vector{Float64}}
    dt::Float64
end

# 1.0
# wall_df_path = "/data2/mkhrenov/Shortish September Beads/2023-09-21_13-41/ProcessAndNodeData.tsv"
# n_rows =  21 #11
# n_cols =  58 #28
# ȳ = 1.0

# 0.1
wall_df_path = "/data2/mkhrenov/Shortish September Beads/2023-09-21_13-29/ProcessAndNodeData.tsv"
hardness_data = read("WAAM Beads - Test job # 14.tsv", DataFrame; delim=" ", header=["HV"; "Y (mm)"; "X (mm)"]) # flip as needed

n_beads = 4
n_rows = 12 #6 
n_cols = 62 #30
ȳd = 0.1
wall_df = DataFrame(File(wall_df_path))

hardness_data = filter(:HV => x -> !any(f -> f(x), (ismissing, isnothing, isnan)), hardness_data)

hv = hardness_data[!, "HV"]
x = hardness_data[!, "X (mm)"]
y = hardness_data[!, "Y (mm)"]
H₀ = 170
H₁ = 400
ΔH = H₁ - H₀

hvsubs = hv[(y.>-33.0).&&(x.>23.0)]

l = 0.5

time_vec = wall_df[:, "Program_Time_s"]
I_vec = wall_df[:, "Current_A"]
V_vec = wall_df[:, "Voltage_V"]
WFS_vec = wall_df[:, "WFS_mm_s"] / 1000
TS_vec = wall_df[:, "TCP_Speed_m_s"]
x_vec = wall_df[:, "X_Position_mm"] / 1000
z_vec = wall_df[:, "Z_Position_mm"] / 1000
layer_num_vec = wall_df[:, "Layer_Number"]
# dt = median(time_vec[2:end] - time_vec[1:end-1])

aus_df = DataFrame(File("./Fraction Transformed.csv"))
T_aus = aus_df[:, "Temperature [K]"]
frac_aus = aus_df[:, "Fraction Transformed"]
ȳ = linear_interpolation(T_aus, frac_aus, extrapolation_bc=Flat())

q(T, A, τ) = exp(τ / T + A)
y₀(T, A, τ) = 1 / (1 + 1 / q(T, A, τ))

mse(A, τ) = sum((frac_aus .- y₀.(T_aus, A, τ)) .^ 2)

model = Model(Ipopt.Optimizer)
@variable(model, A, start = 1e-4)
@variable(model, τ, start = 1e4)
@objective(
    model,
    Min,
    mse(A, τ)
)
optimize!(model)
Ar = value(A)
τr = value(τ)

ẏ(y, T, A1, τ1, A2, τ2) = (1 - y) * exp(-τ1 / T + A1) - y * exp(-τ2 / T + A2)

function rk4_step(Δt, xₖ, uₖ, A1, τ1, A2, τ2)
    k₁ = ẏ(xₖ, uₖ, A1, τ1, A2, τ2)
    k₂ = ẏ(xₖ + Δt * k₁ / 2, uₖ, A1, τ1, A2, τ2)
    k₃ = ẏ(xₖ + Δt * k₂ / 2, uₖ, A1, τ1, A2, τ2)
    k₄ = ẏ(xₖ + Δt * k₃, uₖ, A1, τ1, A2, τ2)

    return xₖ + (1 / 6) * (k₁ + 2k₂ + 2k₃ + k₄) * Δt
end

# function simulate(fd, Nk, U, x₀)
#     x = copy(x₀)
#     for k in 1:(Nk-1)
#         x = fd(x, U[k])
#     end

#     return x
# end

tempv = Vector{Vector{Float64}}()
timev = Vector{Vector{Float64}}()
hardness = Vector{Float64}()
p = Dict()
y_root = [8, 7.5, 37.5, 25]
probe = 2


for bead in 1:n_beads
    temp_df = DataFrame(File("/data2/mkhrenov/Trapezoid Hardness/TrapBead$(bead)Low.csv"))
    hardness_df = DataFrame(File(open("/data2/mkhrenov/Trapezoid Hardness/$(bead) Beads.dfq", enc"UTF-16"), delim=' ', header=["HV", "X", "Y"]))

    temp = temp_df[:, "Cursor $bead [C]"] .+ 273.15
    time = temp_df[:, "reltime"]
    hardy = y_root[bead] .- hardness_df[:, "Y"]
    hardx = hardness_df[:, "X"]
    hv = hardness_df[:, "HV"]
    p[bead] = scatter(hardx, hardy, marker_z=hv, markersize=9, aspect_ratio=1)
    probe_idcs = findall(x -> x == probe, hardy)
    probe_index = argmin(hardx[probe_idcs])
    hardn = hv[probe_idcs][probe_index]

    push!(tempv, temp)
    push!(timev, time)
    push!(hardness, hardn)
end


frac_ref = 1 .- (hardness .- H₀) ./ ΔH
dt = mean(timev[1][2:(end-0)] .- timev[1][1:(end-1)]) / 800#400#200

temp_fine = Vector{Vector{Float64}}()
for bead in 1:n_beads
    intrp = linear_interpolation(timev[bead], tempv[bead], extrapolation_bc=Flat())
    push!(temp_fine, intrp.(range(0.0, timev[bead][end], step=dt)))
end

# function simulate_beads(A1, τ1, A2, τ2)
#     f(y, T) = ẏ(y, T, A1, τ1, A2, τ2)
#     fd(x, u) = rk4_step(f, dt, x, u)
#     frac = zeros(typeof(A1), 4)
#     for bead in 1:n_beads
#         Nks = length(temp_fine[bead])
#         frac[bead] = simulate(fd, Nks, temp_fine[bead], 0.0)
#     end

#     return frac
# end

# function mse_trap(A1, τ1)
#     A2 = A1 - Ar
#     τ2 = τ1 + τr
#     frac = simulate_beads(A1, τ1, A2, τ2)
#     return sum((frac - frac_ref).^2)
# end

# mse_z(z) = mse_trap(z[1], z[2])

function simulate!(X, U, x₀, dt, A1, τ1, A2, τ2)
    Nk = length(X)
    X[1] = x₀

    for k in 1:(Nk-1)
        X[k+1] = rk4_step(dt, X[k], U[k], A1, τ1, A2, τ2)
    end

    return X
end

# REPLACE WITH RAW MATHOPTINT TO REMOVE SYMBOLIC MANIPULATIONS
function final_transform_gradient(dt, X, T, θ)
    fd(y, T, θ) = rk4_step(dt, y, T, θ[1], θ[2], θ[1] - Ar, θ[2] + τr)
    Nk = size(X, 2)
    Nx = size(X, 1)

    ∂f∂θ = ForwardDiff.gradient((θₖ) -> fd(X[1], T[1], θₖ), θ)
    ∂xₖ∂θ = copy(∂f∂θ)
    ∂xₖ₋₁∂θ = copy(∂xₖ∂θ)
    ∂f∂xₖ₋₁ = zero(eltype(θ))# Nx, Nx)
    for k in 2:Nk
        ∂f∂θ = ForwardDiff.jacobian((θₖ) -> fd(X[k-1], T[k-1], θₖ), θ)
        ∂f∂xₖ₋₁ = ForwardDiff.jacobian((xₖ₋₁) -> fd(xₖ₋₁, T[k-1], θ), X[k-1])

        ∂xₖ∂θ .= ∂f∂xₖ₋₁ * ∂xₖ₋₁∂θ .+ ∂f∂θ

        ∂xₖ₋₁∂θ .= ∂xₖ∂θ
    end

    ∂y∂θ = ∂xₖ∂θ#[1, :]
    return ∂y∂θ
end

function final_error(ȳ, Xvec)
    n_beads = length(Xvec)
    ŷ = [Xvec[i][end] for i in 1:n_beads]

    return (ŷ - ȳ)
end

function final_error_gradient(ȳ, dt, X, T, θ)
    Nk = size(X, 2)
    ŷ = X[:, Nk]

    return 2.0 .* (ŷ - ȳ) .* final_transform_gradient(dt, X, T, θ)
end

function final_error_jacobian(ȳ, dt, Xvec, Tvec, θ)
    n_beads = length(Xvec)
    Nθ = length(θ)
    ŷ = [Xvec[i][end] for i in 1:n_beads]
    ∂y∂θ = zeros(eltype(θ), n_beads, Nθ)

    for i in 1:n_beads
        ∂y∂θ[i, :] = final_transform_gradient(dt, Xvec[i], Tvec[i], θ)
    end

    return ∂y∂θ' * (ŷ - ȳ) * 2
end

function final_error_hessian(ȳ, dt, X, T, θ)
    ForwardDiff.jacobian(θi -> final_error_jacobian(ȳ, dt, X, T, θi), θ)
end

function simulate!(Xvec, θ, ȳ, Tvec, dt)
    fd(y, T) = rk4_step(dt, y, T, θ[1], θ[2], θ[1] - Ar, θ[2] + τr)
    n_beads = length(Tvec)

    for j in 1:n_beads
        simulate!(Xvec[j], Tvec[j], 0.0, dt, θ[1], θ[2], θ[1] - Ar, θ[2] + τr)
    end

    e = final_error(ȳ, Xvec)
    J = sum(e .^ 2)

    return J
end


function MOI.eval_objective(prob::SysidProblem, z)
    return simulate!(prob.X, z, frac_ref, temp_fine, prob.dt)
end

function MOI.eval_objective_gradient(prob::SysidProblem, grad_f, z)
    # ForwardDiff.gradient!(grad_f, mse_z, z)
    simulate!(prob.X, z, frac_ref, temp_fine, prob.dt)
    grad_f .= final_error_jacobian(frac_ref, prob.dt, prob.X, temp_fine, z)
end

function MOI.hessian_lagrangian_structure(prob::SysidProblem)
    return [(1, 1), (1, 2), (2, 1), (2, 2)]
end

function MOI.eval_hessian_lagrangian(prob::SysidProblem, H, z, σ, μ)
    h = view(H, :)
    Hm = reshape(h, 2, 2)
    # ForwardDiff.hessian!(Hm, mse_z, z)
    simulate!(prob.X, z, frac_ref, temp_fine, prob.dt)
    Hm .= final_error_hessian(frac_ref, prob.dt, prob.X, temp_fine, z)
    H .*= σ
end


problem = SysidProblem([zeros(length(tf)) for tf in temp_fine], dt)
MOI.features_available(prob::SysidProblem) = [:Grad, :Hess]
MOI.initialize(prob::SysidProblem, features) = nothing
z0 = [-3.0; 1e3]
g0 = zeros(2)
H0 = zeros(4)
@time MOI.eval_objective_gradient(problem, g0, z0)
@time MOI.eval_objective_gradient(problem, g0, z0)
@time MOI.eval_hessian_lagrangian(problem, H0, z0, 1.0, 0.0)
@time MOI.eval_hessian_lagrangian(problem, H0, z0, 1.0, 0.0)


solver = Ipopt.Optimizer()
solver.options["hsllib"] = HSL_jll.libhsl_path
solver.options["linear_solver"] = "ma97"
z = MOI.add_variables(solver, 2)
nlp_bounds = MOI.NLPBoundsPair.([], [])
block_data = MOI.NLPBlockData(nlp_bounds, problem, true)
MOI.set(solver, MOI.VariablePrimalStart(), z[1], -3.0)
MOI.set(solver, MOI.VariablePrimalStart(), z[2], 10^3.2)
MOI.add_constraint(solver, z[2], MOI.GreaterThan(0.0))
MOI.set(solver, MOI.NLPBlock(), block_data)
MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
flush(stdout)
MOI.optimize!(solver)

result = MOI.get(solver, MOI.VariablePrimal(), z)
A1 = result[1]
τ1 = result[2]
A2 = A1 - Ar
τ2 = τ1 + τr


# function df_to_measurement_history(wall_df, num_rows, num_cols)
#     N = num_rows * num_cols
#     time_vec = wall_df[:, "Program_Time_s"]
#     K = length(time_vec)
#     Y = zeros(N, K)
#     i = 1

#     for n in names(wall_df)
#         if !startswith(n, "Node")
#             continue
#         end

#         Y[i, :] .= coalesce.(wall_df[:, n], 0.0) .+ 273.15
#         i += 1
#     end

#     Y = [Y[:, k] for k in 1:K]

#     return Y
# end


# T = df_to_measurement_history(wall_df, n_rows, n_cols)

# # animate_measurement_history(T, dt, n_rows, n_cols, path="animation_measured_sept_$(ȳ).mp4", strid=1, width=1000, height=400, quantity="Temperature (K)", scale=(300, 1800))

# nvox = n_rows * n_cols
# Nk = length(T)

# process = WAAMHardnessPrescribedTemp(nvox, T, exp(A1), τ1, exp(A2), τ2)
# f!(dx, x, u, t, zi) = combined_dynamics!(dx, x, u, process, t, zi, dt*200)

# U = [[] for k in 1:Nk]
# x0 = zeros(nvox)

# X = solve_RK4(f!, x0, U, dt, Nk, 0.0, 0)

# heatmap(reshape(H₀ .+ ΔH .* (1 .- X[end]), n_cols, n_rows)', aspect_ratio=:equal, clim=(140, 300))
# savefig("final_hardness_A$(A)_τ$(τ).png")
# Y_hard = [H₀ .+ ΔH .* (1 .- y) for y in X]
# animate_measurement_history(Y_hard, dt, n_rows, n_cols, path="animation_waam_hardness_alt_$(ȳd).mp4", quantity="Vickers Hardness", scale=(140, 300))#300

# mse_trap(A1, τ1) = MOI.eval_objective(problem, [A1; τ1])
# x, y = -5:1.0:4, 2:0.2:3.4
# xg = x * ones(length(y))'
# yg = ones(length(x)) * y'
# z = mse_trap.(xg, 10 .^ yg)