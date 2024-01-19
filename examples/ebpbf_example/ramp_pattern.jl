using ADDOPT
using CSV, Tables

σ = 1e-3 / 2.355 # 1mm
vₘₐₓ = 4e3 # 4km/s
P = 3e3 # 3kW

l = 0.8e-3 # 0.8mm
n = 75 # 60mm x 60mm
nvox = n * n

tot_time = 100e-3 # 100ms
Δtₘᵢₙ = 1e-6 # 1μs
t = range(0.0, tot_time; step=Δtₘᵢₙ)
Nk = length(t)

# Uniform square power target
u = P / nvox * ones(nvox)
U = collect(Iterators.repeated(u, Nk))

# Generate coordinates
rc(idx) = row_col(n, n, idx)
x = rc.(1:nvox)
x = l .* vcat(collect.(x)'...)
reverse!(x, dims=2)
px = x[:, 1]
pz = x[:, 2]

UP, xtzt, Dt = field_to_spots(t, U, Δtₘᵢₙ, P, vₘₐₓ, px, pz, σ, l; method=:greedy, nf=n ÷ 2)
println("Done.")

CSV.write("scan_strat_ramp.csv", Tables.table(vcat([[x[1]; x[2]; Δtₘᵢₙ]' for x in xtzt]...); header=["X", "Y", "Δt"]))
