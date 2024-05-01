using Interpolations
using LinearAlgebra
using ForwardDiff

function gen_knots(points, Nkb)
    arc_lengths = points[2:end] .- points[1:(end-1)]
    arc_lengths = [norm(d) for d in arc_lengths]
    arc_lengths = [0.0; arc_lengths]
    arc_lengths = cumsum(arc_lengths)

    x = linear_interpolation(arc_lengths, [p[1] for p in points], extrapolation_bc=Flat())
    y = linear_interpolation(arc_lengths, [p[2] for p in points], extrapolation_bc=Flat())
    z = linear_interpolation(arc_lengths, [p[3] for p in points], extrapolation_bc=Flat())

    rs = range(0.0, arc_lengths[end], length=Nkb)

    p̄ = [[x(r); y(r); z(r)] for r in rs]
    return p̄
end

function gen_knots(layers, Nkb, Nkc, Nc)
    p̄ = []

    for c in 1:Nc
        points = layers[c]
        append!(p̄, gen_knots(points, Nkb[c]))

        if c < Nc
            append!(p̄, [p̄[end] for k in 1:Nkc])
        else
            append!(p̄, [points[end] for k in 1:Nkc])
        end
    end

    return p̄
end

function gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=0.0025, l=0.001, x₀=nothing)
    Nkb = length(p̄)
    if isnothing(x₀)
        x₀ = zeros(length(xₙ))
    end
    x = [copy(x₀) for k in 1:Nkb]

    for k in 1:Nkb
        x[k] .= map(h -> clamp(((radius + l / 2) - h) / l, 0.0, 1.0), norm.(eachcol([xₙ'; yₙ'; zₙ'] .- p̄[k])))

        if k > 1
            x[k] .= max.(x[k], x[k-1])
        else
            x[k] .= max.(x[k], x₀)
        end
    end

    return x
end

function gen_torch_ref(p̄, xₙ, yₙ, zₙ; radius=0.0025, l=0.001)
    Nkb = length(p̄)
    x = [zeros(length(xₙ)) for k in 1:Nkb]

    for k in 1:Nkb
        # x[k] .= (l^3 / ((2π)^(3 / 2) * radius^3)) .* exp.((norm.(eachcol([xₙ'; yₙ'; zₙ'] .- p̄[k]))).^2 ./ (-2radius^2))
        # x[k][norm.(eachcol([xₙ'; yₙ'; zₙ'] .- p̄[k])) .> (radius + l/2)] .= 0 
        # x[k] ./= sum(x[k])
        x[k] .= map(h -> clamp(((radius + l / 2) - h) / l, 0.0, 1.0), norm.(eachcol([xₙ'; yₙ'; zₙ'] .- p̄[k])))
        x[k] ./= sum(x[k])
    end

    return x
end