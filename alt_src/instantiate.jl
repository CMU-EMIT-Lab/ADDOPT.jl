using Statistics
using StatsBase
using ProgressMeter
using CUDA

function powerfield_to_sequence(dt, τ, tmin, U)
    dwell = tmin * ceil(30τ / tmin)
    spots_per_step = ceil(Int, dt / dwell)
    u_t = zeros(size(U[1]))
    seq = zeros(Int, spots_per_step * length(U))

    s = 1
    for u in U
        u_t .= u ./ sum(u) .* dt
        u_t *= -1.0

        oldidx = 1
        for spot in 1:spots_per_step
            idx = argmin(u_t)
            u_t[idx] += dwell

            oldidx = idx
            seq[s] = idx
            s += 1
        end

    end

    return seq, dwell
end

point(t, p_i, p_f, τ) = p_f - (p_f - p_i) * exp(-t / τ)

function average!(x_average::V, X::Vector{V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    N = length(X)
    x_average .= 0.0

    for k in 1:N
        x_average .+= X[k]
    end
    x_average ./= N
end

function powerfield_to_sequence_with_traverse(dt, τ, tmin, U::Vector{V}, px, py, l, P, σ; steps_per_spot=5) where {T<:AbstractFloat,V<:AbstractVector{T}}
    Nu = length(U[1])
    nframes = length(U)

    dwell = steps_per_spot * tmin
    spots_per_frame = round(Int, dt / dwell)
    steps_per_frame = spots_per_frame * steps_per_spot
    nspots = spots_per_frame * nframes

    seq = zeros(Int, nspots)

    t = cu(transpose(collect(tmin .* (1:steps_per_spot))))
    spot_x_per_idx = CUDA.zeros(Nu, steps_per_spot)
    spot_y_per_idx = CUDA.zeros(Nu, steps_per_spot)
    powerfield_per_idx = CUDA.zeros(Nu, steps_per_spot, Nu)
    powerfield_established = CUDA.zeros(steps_per_frame * nframes, Nu)
    powerfield_established_sum = CUDA.zeros(1, Nu)
    powerfield_offset = 0
    powerfield_averages = CUDA.zeros(Nu, Nu)
    error = CUDA.zeros(Nu)

    xi, yi = px[1], py[1]
    cx, cy = cu(px), cu(py)
    cx3 = reshape(cx, (1, 1, Nu))
    cy3 = reshape(cy, (1, 1, Nu))

    progress = Progress(nspots)
    for (frame, u) in enumerate(U)
        for spot in ((frame-1)*spots_per_frame+1):(frame*spots_per_frame)
            # Compute trajectories for all possible destinations
            spot_x_per_idx .= @. cx - (cx - xi) * exp(-t / τ)
            spot_y_per_idx .= @. cy - (cy - yi) * exp(-t / τ)

            # Compute powerfields for all trajectories
            powerfield_per_idx .= exp.(((cx3 .- spot_x_per_idx).^2 .+ (cy3 .- spot_y_per_idx).^2) ./ (-2σ^2))
            powerfield_per_idx .*= (l^2 / (2π * σ^2)) * P

            # Add established powerfield history
            sum!(powerfield_established_sum, view(powerfield_established, max(powerfield_offset - 2steps_per_frame + steps_per_spot, 1):(powerfield_offset+1) , :))

            # Compute average powerfield for each destination
            sum!(reshape(powerfield_averages, (Nu, 1, Nu)), powerfield_per_idx)
            powerfield_averages .+= powerfield_established_sum 
            powerfield_averages ./= 2steps_per_frame

            # Compute mean squared error from current reference powerfield for each destination
            powerfield_averages .-= transpose(u)
            powerfield_averages .^= 2
            sum!(error, powerfield_averages)
            error ./= Nu

            # Select minimum, update established powerfield circular array
            next_idx = argmin(error)
            powerfield_established[powerfield_offset .+ (1:steps_per_spot), :] .= view(powerfield_per_idx, next_idx, :, :)

            powerfield_offset += steps_per_spot
            xi, yi = px[next_idx], py[next_idx]
            seq[spot] = next_idx
            next!(progress)
        end
    end

    return seq, dwell, powerfield_established
end