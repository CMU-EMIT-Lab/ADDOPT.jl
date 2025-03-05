include("../ADDOPT.jl")
using Plots
using Profile

Pₘₐₓ = 800.0e-3  # kW
T∞ = 293.15e-3    # K
Tₘₐₓ = 900.0e-3  # K
h = 1.0      # kW/m² kK
A = 1.0       # m²
m = 0.01        # kg
cₚ = 502.416   # kJ/kg kK 

lnA = 38.005
n = 0.051590
E = 240.24e-3 # kJ / mol K

dt = 0.05 # s
Nk = 160

x₀ = [T∞; log(-log(1 - 0.1))]
x̄ = [T∞; log(-log(1 - 0.9))]
ū = [0.0]

furnace = Furnace(h, T∞, m, cₚ, A, dt)
tempering = Tempering(lnA, n, E, dt)
furnace_tempering = FurnaceTempering(furnace, tempering)

nilmx = Matrix{Float64}(undef, 0, 2)
nilmu = Matrix{Float64}(undef, 0, 1)
nilv = Vector{Float64}(undef, 0)
linear_constraint = LinearConstraint(nilmx, nilv, nilmu, nilv, Array(1.0 * I(2)), [Tₘₐₓ; 4.0], collect([1.0 -1.0]'), [Pₘₐₓ; 0.0])

Q = diagm([0.0; 1.0])
R = diagm([1e-4])
quadratic_cost = QuadraticCost(Q, R, x̄, ū)
final_cost = QuadraticCost(10 .* Q, 1 .* R, 1 .* x̄, 1 .* ū)

dynamics = [furnace_tempering for _ in 1:Nk]
constraints = [linear_constraint for _ in 1:Nk]
costs = [quadratic_cost for _ in 1:(Nk-1)]
push!(costs, final_cost)

process = Process(Nk, dynamics, Vector{Float64})
problem = Problem(x₀, process, costs, constraints, Matrix{Float64})
U0 = [[600.0e-3] for _ in 1:(Nk-1)]


@time rollout!(problem, U0)
@time al_ddp!(problem; maxiters=20, ϕ=2.0, gtol=1e-4, verbosity=2)
rollout!(problem, U0)
@time al_ddp!(problem; maxiters=20, ϕ=2.0, gtol=1e-4, verbosity=2)
rollout!(problem, U0)
@profview_allocs al_ddp!(problem; maxiters=20, ϕ=2.0) sample_rate = 0.001
rollout!(problem, U0)
@profview al_ddp!(problem; maxiters=20, ϕ=2.0)

@time rollout!(problem, U0)
@time al_ddp!(problem; maxiters=20, ϕ=2.0, gtol=1e-4, verbosity=2)
X, U = problem.z.X, problem.z.U
t = dt*0:(Nk-1)
T = [x[1] for x in X]
y = [x[2] for x in X]
P = [u[1] for u in U]
plot()
plot!(t, T .* 1e3, label="T")
plot!(t, y .* 100, label="ŷ")
plot!(t, P .* 1e3, label="P")