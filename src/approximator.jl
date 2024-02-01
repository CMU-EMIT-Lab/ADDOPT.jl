using Statistics
using StatsBase
using ProgressMeter

p(t, p_i, p_f, ω) = p_f - (p_f - p_i) * exp(-ω * t)

function field_to_spots(ts, U_ref, Δtₘᵢₙ, P, ω, px, pz, σ, l; method=:random, nf=30, arrival_threshold=0.05)
    tf = ts[end]
    nvox = length(px)
    U, X, Dt, dt = [], [], [], []
    Uin = [zeros(nvox) for i in 1:(nf+1)]
    Dtin = zeros(nf + 1)
    k = 1
    t = 0.0
    pc, pn = l * ones(2), l * ones(2)
    pₘᵢₙ = arrival_threshold * l
    oldidx = 1

    ΣU = zeros(nvox)

    p = Progress(Int(round(tf * 1e6)); dt=0.25, desc="Computing power field approximation... ")
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
            idx = greedy_selection(pc, U_ref[k], Δtₘᵢₙ, px, pz, pₘᵢₙ, l, σ, ω, P, U, Dt, Uin, Dtin, oldidx)
        elseif method == :min
            idx = min_sum_selection(px, pz, U, Dt, ΣU)
        end
        pn .= [px[idx]; pz[idx]]

        dp = pn .- pc
        Δt = ceil((-1 / ω) * log(pₘᵢₙ / norm(dp)) / Δtₘᵢₙ) * Δtₘᵢₙ
        pc, Δt = beam_to!(pc, pn, Δt, px, pz, l, σ, ω, P, Uin, Dtin)
        t += Δt
        append!(Dt, Dtin)
        append!(U, [copy(u) for u in Uin])
        push!(X, copy(pc))
        push!(dt, Δt)
        update!(p, Int(round(t * 1e6)))
        oldidx = idx
        for (u, dt) in zip(Uin, Dtin)
            @. ΣU += u * dt
        end
    end
    update!(p, Int(round(tf * 1e6)))

    return U, X, Dt, dt
end

function random_selection(U_ref)
    return sample(Weights(U_ref / sum(U_ref)))
end

function min_sum_selection(px, pz, U, Dt, ΣU)
    return argmin(ΣU)
end

function greedy_selection(pc, U_ref, Δtₘᵢₙ, px, pz, pₘᵢₙ, l, σ, ω, P, U, Dt, Uin, Dtin, oldidx)
    nvox = length(px)
    Ū = zeros(nvox)
    Û = zeros(nvox)
    err = zeros(nvox)
    err .= Inf
    t_window = nvox * 2 * 1e-6

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
        dp = pn .- pc
        Δt = ceil((-1 / ω) * log(pₘᵢₙ / norm(dp)) / Δtₘᵢₙ) * Δtₘᵢₙ
        beam_to!(pc, pn, Δt, px, pz, l, σ, ω, P, Uin, Dtin)

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
    err[oldidx] = Inf

    return argmin(err)
end

function beam_to!(p0, p1, Δt::Float64, px, pz, l, σ::Float64, ω, P, U, Dt)
    nf = length(Dt)

    Δtf = Δt / nf
    Dt .= Δtf
    ps = [p(i * Δtf, p0, p1, ω) for i in 0:(nf-1)]

    @fastmath @inbounds for (i, pt) in enumerate(ps)
        map!((px, pz) -> power_at_loc(px, pz, pt[1], pt[2], σ), U[i], px, pz)
        U[i] .*= (l^2 / (2π * σ^2)) * P
    end
    map!((px, pz) -> power_at_loc(px, pz, p1[1], p1[2], σ), U[end], px, pz)
    U[end] .*= (l^2 / (2π * σ^2)) * P

    return p(Δt, p0, p1, ω), Δt
end

@inline function power_at_loc(px::Float64, pz::Float64, x::Float64, z::Float64, σ::Float64)::Float64
    return exp(-((px - x)^2 + (pz - z)^2) / (2σ^2))
end

function spots_to_field(xtzt, dts, P, ω, px, pz, σ, l; nf = 30)
    nvox = length(px)
    U, Dt = [], []
    Uin = [zeros(nvox) for _ in 1:(nf+1)]
    Dtin = zeros(nf + 1)
    pc = l * ones(2)

    p = Progress(length(xtzt); dt=0.25, desc="Simulating spot sequence... ")
    for (pn, dt) in zip(xtzt, dts)
        pc, Δt = beam_to!(pc, pn, dt, px, pz, l, σ, ω, P, Uin, Dtin)

        append!(Dt, Dtin)
        append!(U, [copy(u) for u in Uin])
        next!(p)
    end

    return U, Dt
end