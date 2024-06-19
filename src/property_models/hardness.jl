using StatsFuns
using Symbolics
using DataFrames
using Interpolations
using CSV: File

struct HardnessDynamics <: PropertyDynamics
	lnA::Float64 # Constant rate log
	E::Float64 # Activation energy (kJ/mol)
	n::Float64 # Avrami exponent
	R::Float64 # Ideal gas constant # kJ⋅mol^−1⋅K^−1. 
    num::Int

	T_cache::Dict{Tuple{DataType, Int}, Any}
	dictlock::Threads.SpinLock

	function HardnessDynamics(lnA, n, E; num=1, R = 8.3144598e-3)
		return new(lnA, E, n, R, num, Dict{Tuple{DataType, Int}, Any}(), Threads.SpinLock())
	end
end

@inline Nα(pd::HardnessDynamics)::Int = pd.num * 1
property_min(pd::HardnessDynamics) = -Inf * ones(pd.num) # zeros(pd.num)
property_max(pd::HardnessDynamics) = Inf * ones(pd.num)

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
		map!((T, α) -> (T < 950.0 ? (pd.n * exp(pd.lnA - pd.E / pd.R / T) * α^(1 - 1 / pd.n)) : (-log(0.9)-α) ), dα, T, α) # (logistic((1000.0 - T) / 10.0) - α) * 
	end
end
