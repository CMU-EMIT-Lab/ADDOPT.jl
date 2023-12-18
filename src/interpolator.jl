using Statistics
using StatsBase

function field_to_spots(ts, U_ref, Δtₘᵢₙ, P, vₘₐₓ, px, pz, σ, l, mask)
    nvox = length(U_ref[1])
    p_dist = zeros(nvox)
    p_goal = zeros(nvox)
    window_size = 100
    circ_arr = [zeros(nvox) for i in 1:window_size]
    u = zeros(nvox)
    tf = ts[end]
    U = []
    k = 1
    i = 0
    j = 1
    t = 0.0

    while t < tf
        while t > ts[k]
            k += 1
        end
        p_goal .= p_dist ./ U_ref[k]
        p_goal[.!mask] .= Inf

        idx = argmin(p_goal)
        # idx = sample(Weights(U_ref[k] / sum(U_ref[k])))
        xₜ, zₜ = px[idx], pz[idx]
        @. u = (l^2 / (2π * σ^2)) * exp(-((px - xₜ)^2 + (pz - zₜ)^2) / (2σ^2)) * P
        circ_arr[j] .= u
        j = (j + 1) % (window_size+1)
        j = j == 0 ? 1 : j

        p_dist .= mean(circ_arr)
        push!(U, copy(u))
        t += Δtₘᵢₙ
        i += 1
    end

    return U
end