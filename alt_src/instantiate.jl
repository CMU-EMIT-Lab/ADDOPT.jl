function powerfield_to_sequence(dt, τ, tmin, U)
    dwell = tmin * ceil(30τ / tmin)
    n_spots = ceil(Int, dt / dwell)
    u_t = zeros(size(U[1]))
    seq = zeros(Int, n_spots*length(U))

    s = 1
    for u in U
        u_t .= u ./ sum(u) .* dt
        u_t *= -1.0

        oldidx = 1
        for spot in 1:n_spots
            idx = argmin(u_t)
            u_t[idx] += dwell

            oldidx = idx
            seq[s] = idx
            s += 1
        end

    end

    return seq, dwell
end