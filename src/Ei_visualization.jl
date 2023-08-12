using Plots
using JLD2

nx = 25
ny = 4
nz = 4

l = 2e-3
ρ = 7826.0 # kg / m^3
cₚ = 502.416 # J / kg K

ȳ = 0.8

Ei = load_object("ic_waam_Ei_$(ȳ).jld2")
Ei = reshape(Ei, (nx,ny,nz))

Ti = Ei ./ (l^3 * ρ * cₚ)
xi = clamp.(Ei, 0, 1)

Ti = sum(Ti.*xi, dims=2) ./ sum(xi, dims=2)
Ti = reshape(Ti, (nx, nz))'
reverse!(Ti, dims=2)

hm = heatmap((1:nx).*l./1e-3, (1:nz).*l./1e-3, Ti, aspect_ratio=:equal, clim=(300.0, 1800.0), title="Optimized Pre-Cooling Temperature (K), ȳ=$(ȳ)")
plot(hm, size=(1000, 300))
savefig("fig_waam_Ei_$(ȳ).png")