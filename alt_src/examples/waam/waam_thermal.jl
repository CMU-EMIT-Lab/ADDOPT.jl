using OptimalVoronoi


h∞ = 10 # W / m^2 K
hₐᵣ = 500
η = 0.8

k = 34.0        # W / m⋅K
ρ = 7826.0      # kg / m³
cₚ = 502.416    # J / kg⋅K 
α = k / ρ / cₚ

wire_diam = 0.001143 # m, aka 0.045in
wfs = 67.7e-3   # m / s
v = 5.0e-3      # m / s
V̇ = π / 4 * wire_diam^2 * wfs # m³ / s
A = V̇ / v # m²


dt = 0.1       # s
l = v * dt      # m
Fo = α * dt / l^2

T∞ = 295.0      # K
Tₗ = 1784.0     # K, liquidus
Tmax = 3000.0   # K

