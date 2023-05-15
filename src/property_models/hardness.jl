struct HardnessDynamics <: PropertyDynamics
    A
    τ

    n
    T
    Ġ

    function HardnessDynamics(A, τ; n=1)
        T = zeros(n)
        Ġ = zeros(n)
        
        return new(A, τ, n, T, Ġ)
    end
end

Nα(pd::HardnessDynamics) = pd.n * 1
property_min(pd::HardnessDynamics) = zeros(pd.n)
property_max(pd::HardnessDynamics) = ones(pd.n)

function dynamics_function!(pd::HardnessDynamics, td::TransferDynamics, dα, α, s)
    temperature!(td, pd.T, s)
    @. pd.Ġ = pd.A * exp(-pd.τ / pd.T)
    @. dα = (1 - α) * pd.Ġ
end