using Statistics
using StatsBase

function field_to_spots(ts, U_ref, Δt, P, v, px, pz, σ, l; method=:random)
    tf = ts[end]
    nvox = length(px)
    U, X, Dt = [], [], []
    nf = 20
    Uin = [zeros(nvox) for i in 1:(nf+1)]
    Dtin = zeros(nf + 1)
    k = 1
    t = 0.0
    pc, pn = zeros(2), zeros(2)
    oldidx = 1

    p = Progress(Int(round(tf * 1e6)); dt=0.25)
    while t < tf
        while t > ts[k]
            k += 1
        end

        idx = oldidx
        if method == :random
            while idx == oldidx
                idx = random_selection(U_ref[k])
            end
        elseif method == :greedy
            idx = greedy_selection(pc, U_ref[k], Δt, px, pz, l, σ, v, P, U, Dt, Uin, Dtin)
        end
        pn .= [px[idx]; pz[idx]]

        beam_to!(pc, pn, Δt, px, pz, l, σ, v, P, Uin, Dtin)
        pc .= pn
        t += sum(Dtin)
        append!(Dt, Dtin)
        append!(U, [copy(u) for u in Uin])
        push!(X, copy(pc))
        update!(p, Int(round(t * 1e6)))
        oldidx = idx
    end

    return U, X, Dt
end

function random_selection(U_ref)
    return sample(Weights(U_ref / sum(U_ref)))
end

function greedy_selection(pc, U_ref, Δt, px, pz, l, σ, v, P, U, Dt, Uin, Dtin)
    nvox = length(px)
    Ū = zeros(nvox)
    Û = zeros(nvox)
    err = zeros(nvox)
    err .= Inf
    t_window = 100e-6

    t̂ = 0.0
    j = 0
    while t̂ < t_window && j < length(Dt)
        Û .+= U[end-j] .* Dt[end-j]
        t̂ += Dt[end-j]
        j += 1
    end

    candidates = findall(>(1e-4), U_ref)

    for i in candidates
        Ū .= Û
        t̄ = t̂

        pn = [px[i]; pz[i]]
        beam_to!(pc, pn, Δt, px, pz, l, σ, v, P, Uin, Dtin)

        for (u, dt) in zip(Uin, Dtin)
            Ū .+= u .* dt
            t̄ += dt
        end
        Ū ./= t̄

        # Compute squared norm of difference
        Ū .-= U_ref
        Ū .^= 2

        err[i] = sum(Ū)
    end

    return argmin(err)
end

function beam_to!(p0, p1, Δt, px, pz, l, σ, v, P, U, Dt)
    p = copy(p0)
    dp = p1 .- p0
    nf = length(Dt) - 1

    Δtf = norm(dp) / v / nf
    Dt[1:(end-1)] .= Δtf
    Dt[end] = Δt

    for i in 1:nf
        map!((px, pz) -> power_at_loc(px, pz, p[1], p[2], σ), U[i], px, pz)
        U[i] .*= (l^2 / (2π * σ^2)) * P
        @. p += dp / nf
    end
    map!((px, pz) -> power_at_loc(px, pz, p[1], p[2], σ), U[end], px, pz)
    U[end] .*= (l^2 / (2π * σ^2)) * P

end

function power_at_loc(px, pz, x, z, σ)::Float64
    return exp(-((px - x)^2 + (pz - z)^2) / (2σ^2))
end