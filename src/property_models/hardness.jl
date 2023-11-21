using StatsFuns
using Symbolics
using DataFrames
using Interpolations
using CSV: File

struct HardnessDynamics <: PropertyDynamics
    At::Float64 # 1
    τt::Float64
    # Aa::Float64 # 2
    # τa::Float64

    n::Int

    T_cache::Dict{Tuple{DataType,Int},Any}
    dictlock::Threads.SpinLock

    function HardnessDynamics(A1, τ1; n=1) #, A2, τ2    
        return new(A1, τ1, n, Dict{Tuple{DataType,Int},Any}(), Threads.SpinLock())#, A2, τ2,
    end
end

@inline Nα(pd::HardnessDynamics)::Int = pd.n * 1
property_min(pd::HardnessDynamics) = -Inf * ones(pd.n) # zeros(pd.n)
property_max(pd::HardnessDynamics) = Inf * ones(pd.n)

function dynamics_function!(pd::HardnessDynamics, td::TransferDynamics, dα::AbstractVector{Ty}, α, s, t, zi) where {Ty}
    thread = Threads.threadid()
    lock(pd.dictlock)
    T = get!(pd.T_cache, (Ty, thread)) do
        zeros(Ty, Nα(pd))
    end::Vector{Ty}
    unlock(pd.dictlock)

    temperature!(td, T, s, zi)
    if eltype(dα) == Symbolics.Num
        dα .= s .+ α    
    else
        map!((T, α) -> (logistic((1000.0 - T) / 10.0) - α) * pd.At * exp(-pd.τt / T) , dα, T, α) #+ (0 - α) * pd.Aa * exp(-pd.τa / T)
    end
end