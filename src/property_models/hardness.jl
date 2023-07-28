using StatsFuns
using Symbolics

struct HardnessDynamics <: PropertyDynamics
    A::Float64
    τ::Float64

    n::Int
    T_cache::Dict{Tuple{DataType,Int},Any}
    dictlock::Threads.SpinLock

    function HardnessDynamics(A, τ; n=1)


        return new(A, τ, n, Dict{Tuple{DataType,Int},Any}(), Threads.SpinLock())
    end
end

@inline Nα(pd::HardnessDynamics)::Int = pd.n * 1
property_min(pd::HardnessDynamics) = -Inf * ones(pd.n) # zeros(pd.n)
property_max(pd::HardnessDynamics) = Inf * ones(pd.n)

function dynamics_function!(pd::HardnessDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t) where {Ty}
    thread = Threads.threadid()
    lock(pd.dictlock)
    T = get!(pd.T_cache, (Ty, thread)) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}
    unlock(pd.dictlock)

    temperature!(td, T, s)
    @. dα = (1 - α) * pd.A * exp(-pd.τ / T)
    # dα[T .> 1000.0] .= 0.0
    # @. dα = (logistic((1000.0 - T)/10.0) - α) * pd.A * exp(-pd.τ / T) 
    ## @. dα = (1 - α) * pd.A * exp(-pd.τ / T) * logistic((1000.0 - T)/10.0)
    # @. dα = (1 - α) * rate(T)
end

function rate(T)
    if typeof(T) != Symbolics.Num && (T < 800.0 || T > 1100.0)
        return 0.0
    else
        return 3 * ((T - 800) / 100)^3 * ((T - 1100) / 1000)^2
    end
end