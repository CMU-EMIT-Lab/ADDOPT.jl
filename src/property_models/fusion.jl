using Symbolics

struct FusionDynamics <: PropertyDynamics
    T_melt::Float64
    τ::Float64

    n::Int
    T_cache::Dict{DataType, Any}

    function FusionDynamics(T_melt, τ; n=1)
        return new(T_melt, τ, n, Dict{DataType, Any}())
    end
end

@inline Nα(pd::FusionDynamics)::Int = pd.n
property_min(pd::FusionDynamics) = -Inf * ones(pd.n)
property_max(pd::FusionDynamics) = Inf * ones(pd.n)

function dynamics_function!(pd::FusionDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t) where Ty
    T = get!(pd.T_cache, Ty) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}

    temperature!(td, T, s)
    # @. dα = (1 - α) * logistic((T - pd.T_melt)/10) * pd.τ

    if eltype(T) == Symbolics.Num
        @. dα = pd.τ * (T - α)^2
    else
        T .-= α
        clamp!(T, 0.0, Inf)
        T .^= 2
        @. dα = pd.τ * T
    end
    # @. dα = (1 - α) 
    
    # if eltype(T) == Symbolics.Num
    #     @. dα *= pd.τ * (T - pd.T_melt)^2
    # else
    #     T .-= pd.T_melt
    #     clamp!(T, 0.0, Inf)
    #     T .^= 2
    #     @. dα *= pd.τ * T
    # end
end