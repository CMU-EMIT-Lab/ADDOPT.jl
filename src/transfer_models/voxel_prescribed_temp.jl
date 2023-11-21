struct VoxelPrescribedTemperatureDynamics <: TransferDynamics
    Y::Vector{Vector{Float64}}
end

@inline Ns(td::VoxelPrescribedTemperatureDynamics)::Int = 0
state_min(td::VoxelPrescribedTemperatureDynamics) = []
state_max(td::VoxelPrescribedTemperatureDynamics) = []

function dynamics_function!(td::VoxelPrescribedTemperatureDynamics, ds::AbstractVector{Ty}, s, t, zi) where {Ty}

end

function temperature!(td::VoxelPrescribedTemperatureDynamics, T, s, zi)
    T .= td.Y[zi]
end