using StatsFuns

struct FusionDynamics <: PropertyDynamics
    A::Float64
    τ::Float64

    n::Int
    T_cache::Dict{DataType, Any}

    function FusionDynamics(A, τ; n=1)
        return new(A, τ, n, Dict{DataType, Any}())
    end
end

@inline Nα(pd::FusionDynamics)::Int = pd.n
property_min(pd::FusionDynamics) = zeros(pd.n)
property_max(pd::FusionDynamics) = ones(pd.n)

function dynamics_function!(pd::FusionDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t) where Ty
    T = get!(pd.T_cache, Ty) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}

    temperature!(td, T, s)
    @. dα = (1 - α) * logistic((T - 1700)/100) * 2
end