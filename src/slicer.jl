using Interpolations
using LinearAlgebra
using NLsolve
using SpecialFunctions
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
        append!(p̄, gen_knots(points, Nkb))

        if c < Nc
            append!(p̄, [p̄[end] for k in 1:Nkc])
        else
            append!(p̄, [points[end] for k in 1:Nkc])
        end
    end

    return p̄
end

function gen_fill_ref(p̄, xₙ, yₙ, zₙ; radius=0.0025, l=0.001)
    Nkb = length(p̄)
    x = [zeros(length(xₙ)) for k in 1:Nkb]

    for k in 2:Nkb
        # x[k] .= x[k-1]
        # idx = findall(h -> h <= 0.0, )
        # x[k][idx] .= k

        x[k] .= map(h -> clamp(((radius + l / 2) - h) / l, 0.0, 1.0), norm.(eachcol([xₙ'; yₙ'; zₙ'] .- p̄[k])))
        x[k] .= max.(x[k], x[k-1])
    end

    return x
end

function gen_temp_ref(y, T₀, T∞, A, τ)
    g(T) = τ * expinti(-τ / T) + T * exp(-τ / T)
    R = A / log(1 - y) * (g(T∞) - g(T₀))
    return R
end

function gen_objective_weights(fill_ref)
    Nkb = length(fill_ref)
    Nvox = length(fill_ref[1])
    QRs = [Diagonal(zeros(Nvox)) for k in 1:Nkb]

    temp_weights = zeros(Nvox)

    for k in Nkb:-1:1
        temp_weights .= 1e-1
        temp_weights[(fill_ref[Nkb].>0)] .= 1e-1
        temp_weights[(fill_ref[Nkb].==fill_ref[k]).&&(fill_ref[Nkb].>0)] .= 1e1

        QRs[k][diagind(QRs[k])] .= temp_weights
    end

    return QRs
end

function gen_ref(fill_ref, R, Tmax, Th, Tl, T∞, Δt, ρ, l, cₚ)
    Nk = length(fill_ref)
    Nvox = length(fill_ref[1])

    Tref = [zeros(Nvox) for k in 1:Nk]
    QT = [zeros(Nvox) for k in 1:Nk]

    for vox in 1:Nvox
        if fill_ref[Nk] == 0
            continue
        end

        t = 0.0
        for k in 1:Nk
            if fill_ref[Nk][vox] == fill_ref[k][vox]
                T = (Tmax - T∞) * exp(-t / R) + T∞
                Tref[k][vox] = T * ρ * l^3 * cₚ
                t += Δt

                if T ≤ Th && T ≥ Tl
                    QT[k][vox] = 1e2
                else
                    QT[k][vox] = 1e1
                end
            else
                Tref[k][vox] = T∞ * ρ * l^3 * cₚ
                QT[k][vox] = 1e0
            end
        end
    end

    Q = [Diagonal([Q; 1e0 * ones(Nvox)]) for Q in QT]
    x̄ = [[Tref[k]; clamp.(fill_ref[k], 0.0, 1.0)] for k in 1:Nk]

    return x̄, Q
end

function linear2exp(R, Th, Tl, Tmax, T∞)
    Δt = (Th - Tl) / R

    γ₀ = log((Th - T∞) / (Tmax - T∞))
    γ₁ = log((Tl - T∞) / (Tmax - T∞))
    τ = Δt / (γ₀ - γ₁)

    return τ
end

function gen_exponential(y, Th, T∞, A, τ)
    frac(R) = exp2frac(R, Th, T∞, A, τ)
    R = 2.0

    for k in 1:1000
        ŷ = y - frac(R)
        dŷ = ForwardDiff.derivative(R -> y - frac(R), R)
        ΔR = -ŷ / dŷ
        R += ΔR

        if abs(ΔR) < 1e-6
            return R
        end
    end

    println("sad")
    return nothing
end

function exp2frac(R, Th, T∞, A, τ; Δt=0.01, Nk=10000)
    y = 0.0
    ẏ(y, T) = (1 - y) * A * exp(-τ / T)
    T(t) = (Th - T∞) * exp(-t / R) + T∞


    for k in 1:1000
        t = (k - 1) * Δt

        k₁ = ẏ(y, T(t))
        k₂ = ẏ(y + k₁ * Δt / 2, T(t + Δt / 2))
        k₃ = ẏ(y + k₂ * Δt / 2, T(t + Δt / 2))
        k₄ = ẏ(y + k₃ * Δt, T(t + Δt))

        y += (Δt / 6) * (k₁ + 2k₂ + 2k₃ + k₄)
    end

    return y
end