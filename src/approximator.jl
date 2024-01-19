using Statistics
using StatsBase
using ProgressMeter

function field_to_spots(ts, U_ref, Δt, P, v, px, pz, σ, l; method=:random, nf=30)
    tf = ts[end]
    nvox = length(px)
    U, X, Dt = [], [], []
    Uin = [zeros(nvox) for i in 1:(nf+1)]
    Dtin = zeros(nf + 1)
    k = 1
    t = 0.0
    pc, pn = l * ones(2), l * ones(2)
    oldidx = 1

    # Multithreading caches
    Ūth = [zeros(nvox) for n in 1:Threads.nthreads()]
    Uth = [deepcopy(Uin) for n in 1:Threads.nthreads()]
    Dtth = [deepcopy(Dtin) for n in 1:Threads.nthreads()]

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
            idx = greedy_selection(pc, U_ref[k], Δt, px, pz, l, σ, v, P, U, Dt, Uin, Dtin, oldidx, Ūth, Uth, Dtth)
        elseif method == :min
            idx = min_sum_selection(px, pz, U, Dt, ΣU)
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
        for (u, dt) in zip(Uin, Dtin)
            @. ΣU += u * dt
        end
    end

    return U, X, Dt
end

function random_selection(U_ref)
    return sample(Weights(U_ref / sum(U_ref)))
end

function min_sum_selection(px, pz, U, Dt, ΣU)
    return argmin(ΣU)
end

function greedy_selection(pc, U_ref, Δt, px, pz, l, σ, v, P, U, Dt, Uin, Dtin, oldidx, Ūth, Uth, Dtth)
    nvox = length(px)
    Ū = zeros(nvox)
    Û = zeros(nvox)
    err = zeros(nvox)
    err .= Inf
    t_window = nvox / 3 * 1e-6

    t̂ = 0.0
    j = 0
    while t̂ < t_window && j < length(Dt)
        Û .+= U[end-j] .* Dt[end-j]
        t̂ += Dt[end-j]
        j += 1
    end

    candidates = findall(>(1e-4), U_ref)

    Threads.@threads :static for i in candidates
        # Multithreading cache selection
        thread = Threads.threadid()
        Ū = Ūth[thread]
        U = Uth[thread]
        Dt = Dtth[thread]
        
        Ū .= Û
        t̄ = t̂

        pn = [px[i]; pz[i]]
        # beam_to!(pc, pn, Δt, px, pz, l, σ, v, P, Uin, Dtin)
        beam_to!(pc, pn, Δt, px, pz, l, σ, v, P, U, Dt)

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

function beam_to!(p0, p1, Δt, px, pz, l, σ::Float64, v, P, U, Dt)
    nf = length(Dt) - 1
    dp = (p1 .- p0) / nf
    ps = [p0 .+ dp * i for i in 1:nf]

    Δtf = norm(dp) / v
    Dt[1:(end-1)] .= Δtf
    Dt[end] = Δt

    @fastmath @inbounds for (i, pt) in enumerate(ps)
        map!((px, pz) -> power_at_loc(px, pz, pt[1], pt[2], σ), U[i], px, pz)
        U[i] .*= (l^2 / (2π * σ^2)) * P
    end
    map!((px, pz) -> power_at_loc(px, pz, p1[1], p1[2], σ), U[end], px, pz)
    U[end] .*= (l^2 / (2π * σ^2)) * P
end

@inline function power_at_loc(px::Float64, pz::Float64, x::Float64, z::Float64, σ::Float64)::Float64
    return exp(-((px - x)^2 + (pz - z)^2) / (2σ^2))
end