using StatsFuns

struct HardnessDynamics <: PropertyDynamics
    A::Float64
    τ::Float64

    n::Int
    T_cache::Dict{DataType, Any}

    function HardnessDynamics(A, τ; n=1)


        return new(A, τ, n, Dict{DataType, Any}())
    end
end

@inline Nα(pd::HardnessDynamics)::Int = pd.n * 1
property_min(pd::HardnessDynamics) = zeros(pd.n)
property_max(pd::HardnessDynamics) = ones(pd.n)

function dynamics_function!(pd::HardnessDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t) where Ty
    T = get!(pd.T_cache, Ty) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}

    temperature!(td, T, s)
    @. dα = (1 - α) * pd.A * exp(-pd.τ / T) #logistic(1000 - T)
end