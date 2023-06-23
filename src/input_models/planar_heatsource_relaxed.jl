using LinearAlgebra

struct PlanarHeatsourceDynamics <: InputDynamics
    nrows::Int
    ncols::Int

    l::Float64
    xₙ::Vector{Float64}
    zₙ::Vector{Float64}

    Pₘₐₓ::Float64
    Pₘᵢₙ::Float64

    σ::Float64

    ρ::Float64
    cₚ::Float64

    Qx::Matrix{Float64}
    Qz::Matrix{Float64}

    Qvx::Matrix{Float64}
    Qvz::Matrix{Float64}

    function PlanarHeatsourceDynamics(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, ρ, cₚ, σ, dmax)
        nvox = nrows * ncols
        dmax_x = dmax_z = dmax

        Qx = ones(nvox) * (xₙ .^ 2)' - (xₙ * xₙ') - σ^2 * ones(nvox, nvox)
        Qz = ones(nvox) * (zₙ .^ 2)' - (zₙ * zₙ') - σ^2 * ones(nvox, nvox)

        Qvx = xₙ * ones(nvox)' - ones(nvox) * xₙ' - dmax_x * ones(nvox, nvox)
        Qvz = zₙ * ones(nvox)' - ones(nvox) * zₙ' - dmax_z * ones(nvox, nvox)

        return new(nrows, ncols, l, xₙ, zₙ, Pₘₐₓ, Pₘᵢₙ, σ, ρ, cₚ, 20Qx, 20Qz, Qvx, Qvz)
    end
end

Nu(id::PlanarHeatsourceDynamics)::Int = id.nrows * id.ncols # P (W for power)
Nr(id::PlanarHeatsourceDynamics)::Int = 0

Nc_ineq(id::PlanarHeatsourceDynamics)::Int = 3
Nc_eq(id::PlanarHeatsourceDynamics)::Int = 0
Nc_ineq_inter(id::PlanarHeatsourceDynamics)::Int = 4

ineq_min(id::PlanarHeatsourceDynamics) = [id.Pₘᵢₙ; -Inf; -Inf]
ineq_max(id::PlanarHeatsourceDynamics) = [id.Pₘₐₓ; 0.0; 0.0]

ineq_inter_min(id::PlanarHeatsourceDynamics) = [-Inf; -Inf; -Inf; -Inf]
ineq_inter_max(id::PlanarHeatsourceDynamics) = [0.0; 0.0; 0.0; 0.0]

input_min(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)
input_max(id::PlanarHeatsourceDynamics) = Inf * ones(id.nrows * id.ncols)
input_idle(id::PlanarHeatsourceDynamics) = zeros(id.nrows * id.ncols)

state_min(id::PlanarHeatsourceDynamics) = []
state_max(id::PlanarHeatsourceDynamics) = []

function dynamics_function!(id::PlanarHeatsourceDynamics, dr::AbstractVector{Ty}, s, r, u, t) where {Ty}

end

function input_function!(id::PlanarHeatsourceDynamics, ds::AbstractVector{Ty}, r, u, t) where {Ty}
    dT = ds
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    ρ, cₚ = id.ρ, id.cₚ

    # Forced / input dynamics
    @. dT += u / (ρ * l^3 * cₚ)
end

function equality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, u, t) where {Ty}

end

function inequality_constraint!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, u, t) where {Ty}
    l, xₙ, zₙ, σ = id.l, id.xₙ, id.zₙ, id.σ
    nvox = id.nrows * id.ncols
    Qx, Qz = id.Qx, id.Qz

    c[1] = sum(u)
    c[2] = 0.5 * dot(u, Qx, u)
    c[3] = 0.5 * dot(u, Qz, u)
end

function inequality_constraint_interstep!(id::PlanarHeatsourceDynamics, c::AbstractVector{Ty}, uₖ, t, uₖ₊₁) where {Ty}
    Qvx, Qvz = id.Qvx, id.Qvz

    c[1] = dot(uₖ, Qvx, uₖ₊₁)
    c[2] = dot(uₖ₊₁, Qvx, uₖ)
    c[3] = dot(uₖ, Qvz, uₖ₊₁)
    c[4] = dot(uₖ₊₁, Qvz, uₖ)
end

function col_row(col, row)
    ret = Vector{Tuple{Int,Int}}()
    for c in 1:col
        for r in 1:row
            push!(ret, (r, c))
        end
    end

    return ret
end

function equality_constraint_hessian_structure(id::PlanarHeatsourceDynamics, i)
    return []
end

function equality_constraint_hessian_values(id::PlanarHeatsourceDynamics, H::AbstractVector{Ty}, μ, i, j, k) where {Ty}
    j += 1
    return i, j
end

function inequality_constraint_hessian_structure(id::PlanarHeatsourceDynamics, i)
    if i == 1
        return []
    elseif i == 2
        return col_row(size(id.Qx)...)
    elseif i == 3
        return col_row(size(id.Qz)...)
    end
end

function inequality_constraint_hessian_values(id::PlanarHeatsourceDynamics, H::AbstractVector{Ty}, μ, i, j, k) where {Ty}
    if k == 1
    elseif k == 2
        Qx_val = vec(id.Qx)
        H[i:(i+length(Qx_val)-1)] .= Qx_val .* μ[j]
        i += length(Qx_val)
    elseif k == 3
        Qz_val = vec(id.Qz)
        H[i:(i+length(Qz_val)-1)] .= Qz_val .* μ[j]
        i += length(Qz_val)
    end
    j += 1

    return i, j
end

function inequality_interstep_constraint_hessian_structure(id::PlanarHeatsourceDynamics, i)
    box = col_row(id.nrows * id.ncols, id.nrows * id.ncols)

    box1 = [(row, col + Nu(id)) for (row, col) in box]
    box2 = [(row + Nu(id), col) for (row, col) in box]

    return vcat(box1, box2)
end

function inequality_interstep_constraint_hessian_values(id::PlanarHeatsourceDynamics, H::AbstractVector{Ty}, μ, i, j, k) where {Ty}
    Qvx = vec(id.Qvx)
    Qvz = vec(id.Qvz)
    Qv = k < 3 ? Qvx : Qvz
    l = length(Qv)

    H[i:(i+l-1)] .= Qv .* μ[j]
    i += l
    H[i:(i+l-1)] .= Qv .* μ[j]
    i += l
    j += 1

    return i, j
end