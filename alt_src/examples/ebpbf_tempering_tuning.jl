include("../ADDOPT.jl")
using Plots
using Profile
using LinearAlgebra
using Images
using CUDA
using FiniteDiff
using JLD2
using Optim, NLSolversBase
import NLSolversBase: clear!
using ImageTransformations, CoordinateTransformations

# Type setup
Ty = Float32
V = CuVector{Ty}
M = CuMatrix{Ty}

######## Functions ########

function image_error(img, img_ref, dx, dy)
    Nx, Ny = size(img)
    img_off = warp(img_ref, Translation(dx, dy))
    return norm(img .- view(img_off, 1:Nx, 1:Ny))
end

function align_images(img, img_ref; step=0.25)
    Nx, Ny = size(img)
    Nx_ref, Ny_ref = size(img_ref)
    dx = 0:step:(Nx_ref-Nx)
    dy = 0:step:(Ny_ref-Ny)
    dx_grid = ones(length(dy)) * dx'
    dy_grid = dy * ones(length(dx))'

    error(dx, dy) = image_error(img, img_ref, dx, dy)
    error_arr = error.(dx_grid, dy_grid)
    idx = argmin(error_arr)
    dx_opt, dy_opt = dx_grid[idx], dy_grid[idx]

    return warp(img_ref, Translation(dx_opt, dy_opt))
end

function simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞)
    tempering = Tempering(lnA, n, E, dt; num=nvox)
    dynamics = PBFTempering(pbf_powerfield, tempering)


    y_init = (HVmax - HV_init) / (HVmax - HVmin)
    x₀ = V([T∞ * ones(nvox); log(-log(1 - y_init)) * ones(nvox)])
    x1, x2 = copy(x₀), copy(x₀)
    for u in U
        transition!(dynamics, x2, x1, η .* u)
        x1 .= x2
    end

    ŷ_end = reverse(reshape(Array(x1[(nvox+1):(nvox+nsvox)]), Nx, Ny), dims=1)
    y_end = 1 .- exp.(-exp.(ŷ_end))
    HV_final = HVmax .- (HVmax - HVmin) * y_end

    return HV_final
end


######## Begin tuning ########

# Geometric parameters
HV_mask = reverse(.!Bool.(Gray.(load("scotty_mask_subs5.png"))), dims=1)
Nx, Ny = size(HV_mask)
Nz = 4
buffer = 2
nsvox = Ny * Nx
nvox = nsvox * Nz
Nu = (Nx - 2buffer) * (Ny - 2buffer)

plot_hardness(HV) = heatmap(HV, clim=(140, 420), cbar_title="HV", aspect_ratio=:equal, colorbar=false, showaxis=false, size=(600, 600), c=:inferno)
plot_error(HV, HV_ref) = heatmap((HV .- HV_ref) .* HV_mask, clim=(-100, 100), cbar_title="HV Error", aspect_ratio=:equal, colorbar=false, showaxis=false, size=(600, 600), c=:berlin)


l = 1e-3 * 5 / 7 # m
η = 1.0f0
σ = 1200e-6 / 1.35 # Spot diameter, m
σ = 1400e-6 / 1.35 # Spot diameter, m

ω = 2π * 178.446e3 # 1/s
Pₛₑₜ = 3000.0 * η # W

HVmin = 140.0f0
HVmax = 400.0f0

# Material parameters, taken at the solidus
k = 31.1e3 # W / kK 
ρ = 7269.0 # kg / m^3 
cₚ = 720.0e3 # J / kg kK 

# Tempering parameters
lnA::Ty = 38.005f0
n::Ty = 0.051590f0
E::Ty = 240.24f-3 # kJ / mol K

# Environment parameters
T∞ = (20.0 + Pₛₑₜ * 0.100 * 50 / ((1 / 2) * (0.1)^2 * sqrt(k / ρ / cₚ * 5.0) * ρ * cₚ) + 273.15) * 1e-3 # kK
T₀ = T∞
h = 0.0 # W / m^2 K (Vacuum)
dt::Ty = 50e-3 # s, aka 50ms

pbf_powerfield = PBFPowerField(Nx, Ny, Nz, l, k, ρ, cₚ, T∞, T₀, h, dt, Ty, V, M; σ=σ, buffer=buffer)

data = [
    (360.0f0, 1.0f0, load_object("initialguess_U_5.jld2"), load_object("HV_ref_naive.jld2")),
    (380.0f0, 1.0f0, load_object("solution_U_old_5.jld2"), load_object("HV_ref_opt.jld2")),
    (370.0f0, 1.0f0, load_object("initialguess_U_5.jld2"), load_object("HV_ref_naive2.jld2")),
    (360.0f0, 1.0f0, load_object("solution_U_new_5.jld2"), load_object("HV_ref_opt2.jld2")),
]

HV_init = [x[1] for x in data]
weights = [x[2] for x in data]
U_arr = [[V(y) for y in x[3]] for x in data]
HV_arr = [x[4] for x in data]
HV_ref = []

for (HV_init, U, HV, HVf1) in zip(HV_init, U_arr, HV_arr, HVf_f1)
    HVf = simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞)
    HVr = align_images(HVf, HV)[1:Nx, 1:Ny] # HVf1
    push!(HV_ref, HVr)
end

# [n * lnA, 1 / n, n * E / 1e3]

θ₀ = Float32.([n * lnA; 1 / n; n * E; HVmax; HVmin; 0.95])

function loss(θ, HV_init, U, HV_ref, HV_mask, dt, nvox, Nx, Ny, T∞)
    nlnA, invn, nE, HVmax, HVmin, η = θ
    n = 1 / invn
    lnA = nlnA / n
    E = nE / n
    J = 0.0f0
    for (HV_init, U, HV_ref) in zip(HV_init, U, HV_ref)
        HVf = simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞)
        J += norm((HVf .- HV_ref) .* HV_mask)^2 / sum(HV_mask) / 100.0
    end

    return J
end
J(θ) = loss(θ, HV_init, U_arr, HV_ref, HV_mask, dt, nvox, Nx, Ny, T∞)

function J_grad!(g, θ)
    FiniteDiff.finite_difference_gradient!(g, J, θ)
    # g .= Zygote.gradient(J, θ)
end
function J_hess!(H, θ)
    FiniteDiff.finite_difference_hessian!(H, J, θ)
end
df = TwiceDifferentiable(J, J_grad!, J_hess!, θ₀)

lower_bounds = Float32.([-Inf; 0.5; 0.0; 390.0; 130.0; 0.6])
upper_bounds = Float32.([Inf; 1000.0; n*300.0f-3; 440.0; 200.0; 1.0])
bounds = TwiceDifferentiableConstraints(lower_bounds, upper_bounds)

nlnA, invn, nE, HVmax, HVmin, η = θ₀
n = 1 / invn
lnA = nlnA / n
E = nE / n

HVf_0 = [simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞) for (U, HV_init) in zip(U_arr, HV_init)]

@show J(θ₀)

using LineSearches

Jc = Inf
θ = copy(θ₀)
inner_optimizer = BFGS(linesearch=LineSearches.BackTracking(order=3))
for _ in 1:(2^6)
    # try
    θ₀ = Float32.([lnA * n * 2 * rand(); 1 / rand(); n * (0.15 * rand() + 0.15); HVmax; HVmin; 0.6 + rand() * 0.4])
    clear!(df)
    res = optimize(J, J_grad!, lower_bounds, upper_bounds, θ₀, Fminbox(inner_optimizer), Optim.Options(show_trace=true, iterations=100))
    # res = optimize(df, bounds, θ₀, IPNewton(), Optim.Options(show_trace=true, iterations=1000))
    if res.minimum < Jc
        @show θ = res.minimizer
        @show Jc = res.minimum
    end
    # catch
    # end
end

nlnA, invn, nE, HVmax, HVmin, η = θ1
n = 1 / invn
lnA = nlnA / n
E = nE / n

HVf_f1 = [simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞) for (U, HV_init) in zip(U_arr, HV_init)]


nlnA, invn, nE, HVmax, HVmin, η = θ
n = 1 / invn
lnA = nlnA / n
E = nE / n

HVf_f = [simulate_process(lnA, n, E, HVmax, HVmin, η, pbf_powerfield, U, HV_init, dt, nvox, Nx, Ny, T∞) for (U, HV_init) in zip(U_arr, HV_init)]

p1 = plot(plot_hardness.(HV_ref)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
p2 = plot(plot_hardness.(HVf_0)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
p3 = plot(plot_hardness.(HVf_f1)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
p4 = plot(plot_hardness.(HVf_f)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
ph = plot(p1, p2, p3, p4, layout=(4, 1), size=(length(U_arr) * 550, 4 * 600))



pe1 = plot(plot_error.(HVf_0, HV_ref)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
pe2 = plot(plot_error.(HVf_f1, HV_ref)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
pe3 = plot(plot_error.(HVf_f, HV_ref)..., layout=(1, length(U_arr)), size=(length(U_arr) * 600, 600))
pe = plot(pe1, pe2, pe3, layout=(3, 1), size=(length(U_arr) * 550, 3 * 600))

@show J(θ)
@show lnA, n, E, HVmax, HVmin, η
    # (8.648193f0, 0.7761927f0, 0.045957092f0, 391.70642f0, 199.9739f0, 0.9365894f0)
θ1 = Ty.([0.358 * 8.845, 1 / 0.358, 0.358 * 56.277e-3, 404.0, 135.0, 1.0])
@show J(θ1)