using Statistics
using StatsBase

function field_to_spots(ts, U_ref, Δtₘᵢₙ, P, vₘₐₓ, px, pz, σ, l, mask)
    nvox = length(U_ref[1])
    u = zeros(nvox)
    tf = ts[end]
    U = []
    k = 1
    t = 0.0

    while t < tf
        while t > ts[k]
            k += 1
        end

        idx = sample(Weights(U_ref[k] / sum(U_ref[k])))
        xₜ, zₜ = px[idx], pz[idx]
        @. u = (l^2 / (2π * σ^2)) * exp(-((px - xₜ)^2 + (pz - zₜ)^2) / (2σ^2)) * P
        
        push!(U, copy(u))
        t += Δtₘᵢₙ
    end

    return U
end