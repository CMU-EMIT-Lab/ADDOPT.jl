struct HardnessDynamics <: PropertyDynamics
    A::Float64
    τ::Float64

    n::Int
    T_cache::Dict{DataType, Any}
    # Ġ

    function HardnessDynamics(A, τ; n=1)


        return new(A, τ, n, Dict{DataType, Any}())
    end
end

@inline Nα(pd::HardnessDynamics)::Int = pd.n * 1
property_min(pd::HardnessDynamics) = zeros(pd.n)
property_max(pd::HardnessDynamics) = ones(pd.n)

function dynamics_function!(pd::HardnessDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s) where Ty
    # T = zeros(eltype(dα), pd.n)
    T = get!(pd.T_cache, Ty) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}
    # Ġ = zeros(eltype(dα), pd.n)

    temperature!(td, T, s)
    # T = s[1]
# @.
    # Ġ = pd.A * exp(-pd.τ / T)
    # @show typeof(dα)
    # @show typeof(α)
    # @show typeof(T)
    @. dα = (1 - α) * pd.A * exp(-pd.τ / T)
    # dα[1] = (1 - α[1]) * Ġ
end