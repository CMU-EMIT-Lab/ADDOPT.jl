using ADDOPT
using JLD2
using CUDA
using LinearAlgebra
using Images
using ProgressMeter
using Plots

include("binary_modeler.jl")

device!(1) # Choose GPU to use
example_path = "examples/lpbf/in718"


y_init = 0.001

# Type setup (scalar, vector, and matrix data types)
Ty = Float32
V = CuVector{Ty} # Cu = CUDA, 32 bit floats for speed
M = CuMatrix{Ty}
D = Dynamics{Ty}

mask_img = Gray.(load("$example_path/invertedpyramid52.png"))
n_subsample = 23 # 18 gives a grid where each voxel is 1mm
mask_img_subsampled = mask_img[(n_subsample÷2):n_subsample:end, (n_subsample÷2):n_subsample:end]
boolean_mesh = Int.(mask_img_subsampled .< 0.5)

# Calculating the dimensions of each voxel based on known dimensions from invertedpyramid.png
vertical_dim = 55 # mm 
horizontal_dim = 37 # mm # 30 for all pyrimads except 52deg, which is 37
(nz, nr) = size(boolean_mesh)
Δr = Ty(horizontal_dim / nr * 1e-3) # m width of voxels
Δz = Ty(vertical_dim / nz * 1e-3) # m height of voxels

# Add a powder buffer to the outside of the part
powder_buffer = 0.012 # m of powder to put on the outside of the part
nBuffer = Int(round(powder_buffer / Δr))
buffer_of_powder = Int.(zeros(nz, nBuffer))
boolean_mesh = hcat(boolean_mesh, buffer_of_powder)
@show (nz, nr) = size(boolean_mesh) # Adjusting dimensions

save_object("BooleanMesh.jld2", boolean_mesh) # Saving boolean mesh with powder buffer for use in some other functions (model validation and cylindrical animation)

# Working to make a list of layers
layer_booleans = Vector{Matrix{Bool}}(undef, nz)
for i = 0:(nz-1)
    if i < 2
        layer_booleans[i+1] = boolean_mesh[(end-1):end, :]
    end
    if i >= 2
        layer_booleans[i+1] = boolean_mesh[(end-i):end, :]
    end
end

# Inconel718 Material parameters, taken at 500 C (773 K), from Mills2002 spreadsheet
conductivity = 18.7 # kW / kK 
density = 8001.0 # kg / m^3 
constant_cₚ = 527.0 # kJ / kg kK 
conductivity_reduction_factor = 0.01
packing_factor = 0.5

# Property vectors for solid and powder sections of domain for each layer
k = Vector{V}(undef, nz)
ρ = Vector{V}(undef, nz)
cₚ = Vector{V}(undef, nz)
for i = 1:nz
    k[i] = V(boolean_vector(layer_booleans[i], conductivity, conductivity_reduction_factor))
    ρ[i] = V(boolean_vector(layer_booleans[i], density, packing_factor))
    cₚ[i] = V(boolean_vector(layer_booleans[i], constant_cₚ, packing_factor))
end

# Beam parameters
η = 0.35 # Absorptivity
P_set::Ty = 285.0 * η * 1e-3 # kW Applied power including Absorptivity

# Tempering parameters (avrami fits)
A::Ty = 1.1389870188240705e8 # - 2.4738e7
lnA::Ty = log(A)
n::Ty = 0.6112108863026963 #- 0.011228
E::Ty = 201.94448680539028e-3 #- 3*1.6603 # kJ / mol 
R = 8.3144598e-3

# Environment parameters
T∞ = 300e-3 #kK
T_base::Ty = (150+273) * 1e-3  #kK Assuming constant baseplate temperature
h::Ty = 200 # kW/m2kK

### CONSTRUCT DYNAMIMCS ###
# Calculating the required number of time steps for each layer in the printing process
scan_speed = 960 # mm/s 
hatch_spacing = 0.110 # mm
layer_thickness = 0.040e-3 # m
lumped_layers = Δz/layer_thickness
build_rate = scan_speed * hatch_spacing # mm^2/s
powered_time_steps = zeros(Int64, nz) # Memory pre-allocation
layer_time = zeros(Float32, nz)

for i = 1:nz # Calculating how long it takes to scan each layer, and multiplying by the number of lumped layers
    radius = count((boolean_mesh[nz-i+1, :]) .> 0) * Δr * 1e3 # mm
    area = π * radius^2
    layer_time[i] = area / build_rate * lumped_layers # Length to scan each layer in the simulation domain
end

# How long the process pauses between each layer
cooling_time = 7*lumped_layers # s, multiply by number of lumped layers

# Post process cooling time
post_process_time = 5000 # s
post_process_steps = 60
post_build_dt::Ty = post_process_time/post_process_steps

total_time_hrs = (sum(layer_time[:]) + cooling_time * nz + post_process_time)/3600


# Set the dt value for the "ground truth" simulation where the temperature histories were generated from
dt_in_GT_sim = 0.5

powered_T_dist = load_object("$example_path/52_deg_Powered_Max_Temperatures_nz_44.jld2")
cooling_T_dist = load_object("$example_path/52_deg_Unpowered_Max_Temperatures_nz_44.jld2")

# Set these initial and final transformed transformation percentages 
init = log(-log(1 - 0.3))
fin = log(-log(1 - 0.4)) # 0.35
max_step = 200 # seconds # 100 
constant_Δt = 50

powered_steps = Vector{}(undef, nz)
cooling_steps = Vector{}(undef, nz)

# Calculate time steps based on groudn truth temperature history predictions
for i in 1:nz
    cooling_steps[i] = fill(0.0f0, 0)
    powered_steps[i] = fill(0.0f0, 0)

    while sum(cooling_steps[i]) < cooling_time
        if isempty(cooling_steps[i])
            push!(cooling_steps[i], minimum([max_step, clamp((exp(fin/n) - exp(init/n))/(A*exp(-E/(R*cooling_T_dist[i][1]))), 0, cooling_time)]))
            # push!(cooling_steps[i], constant_Δt)
        end

        next_step = (exp(fin/n) - exp(init/n)) / (A * exp(-E / (R * cooling_T_dist[i][clamp(Int(floor(sum(cooling_steps[i])/dt_in_GT_sim)), 1, length(cooling_T_dist[i]))])))
        # next_step = constant_Δt # Set for constant time step

        if abs(sum(cooling_steps[i]) - cooling_time) < 0.001 # Checks for total time for this layer being equal to the required time with a small buffer to correct for rounding errors.
            break
            # elseif next_step > cooling_time - sum(cooling_steps[i])
        elseif minimum([max_step, next_step]) > cooling_time - sum(cooling_steps[i])
            push!(cooling_steps[i], cooling_time - sum(cooling_steps[i]))
            break
        else
            # push!(cooling_steps[i], next_step)
            push!(cooling_steps[i], minimum([max_step, next_step]))
        end
    end

    while sum(powered_steps[i]) < layer_time[i]
        if isempty(powered_steps[i])
            push!(powered_steps[i], minimum([max_step, clamp((exp(fin/n) - exp(init/n))/(A*exp(-E/(R*powered_T_dist[i][end]))), 0, layer_time[i])]))
            # push!(powered_steps[i], constant_Δt)
        end

        next_step = (exp(fin/n) - exp(init/n)) / (A * exp(-E / (R * powered_T_dist[i][length(powered_T_dist[i])-clamp(Int(floor(sum(powered_steps[i])/dt_in_GT_sim)), 1, length(cooling_T_dist[i]))])))
        # next_step = constant_Δt # Set for constant time step

        if abs(sum(powered_steps[i]) - layer_time[i]) < 0.001 # Checks for total time for this layer being equal to the required time with a small buffer to correct for rounding errors.
            break
            # elseif next_step > layer_time[i] - sum(powered_steps[i])
        elseif minimum([max_step, next_step]) > layer_time[i] - sum(powered_steps[i])
            push!(powered_steps[i], layer_time[i] - sum(powered_steps[i]))
            break
        else
            # push!(powered_steps[i], next_step)
            push!(powered_steps[i], minimum([max_step, next_step]))
        end
    end

    powered_steps[i] = reverse(powered_steps[i])

end


total_steps = (nz-1) + sum([length(x) for x in powered_steps]) + sum([length(x) for x in cooling_steps]) + post_process_steps
@show total_steps
@show total_time_hrs

println("Initializing Dynamics")

# Memory Pre-allocation
dynamics = Vector{D}(undef, total_steps)
U = Vector{V}(undef, total_steps)
nzs = [size(x, 1) for x in layer_booleans]
buffers = [[findfirst(x[1, :])-1 nr-findlast(x[1, :])] for x in layer_booleans]
dts = Vector{}(undef, total_steps)
powered_locs = zeros(total_steps)

counter = 1 # Keep track of the current time step within the dynamics construction
h_test = 0


effort = cumsum(map(i -> (nzs[i] * nr)^3, 1:nz))
progress = Progress(effort[end], desc="Preparing layer dynamics matrices...")

for i in 1:nz
    if i == 1
        global x₀ = V([T∞*ones(nr*nzs[i]); P_set; T_base; log(-log(1-y_init))*ones(nr*nzs[i])]) # Initial conditions
    end

    # Add layer 
    if i > 1
        temp_dynamics = PBFPowerFieldCylindrical(nzs[i], nr, Δr, Δz, k[i], ρ[i], cₚ[i], T∞, h, 1f0, Ty, V, M; buffer=buffers[i])
        temp_tempering = Tempering(lnA, n, E, 1f0; num=nr*nzs[i])
        dynamics[counter] = PBFTempering_cylindrical(temp_dynamics, temp_tempering)
        U[counter] = V([P_set/1f0*1e-1; 0]) # Scale input here by counter in voxelized_conduction
        dts[counter] = 1f0
        global counter += 1
        global h_test += 1*h
    end

    # Applied Power
    for (j, step_time) in enumerate(powered_steps[i])
        temp_dynamics = PBFPowerFieldCylindrical(nzs[i], nr, Δr, Δz, k[i], ρ[i], cₚ[i], T∞, h/6, Ty(step_time), Ty, V, M; buffer=buffers[i])
        temp_tempering = Tempering(lnA, n, E, Ty(step_time); num=nr*nzs[i])
        dynamics[counter] = PBFTempering_cylindrical(temp_dynamics, temp_tempering)
        if j == length(powered_steps[i])
            U[counter] = V([-P_set/step_time*1e-1; 0]) # Scale input here by counter in voxelized_conduction
        else
            U[counter] = V([0; 0])
        end
        dts[counter] = step_time
        powered_locs[counter] = 1
        global counter += 1
        global h_test += step_time*h/6
    end

    # No Applied Power
    for (j, step_time) in enumerate(cooling_steps[i])
        if j == length(cooling_steps[i]) && i > 1 && i < nz # End layer should be applied to the end of every domain set where the following stack has an additional layer
            temp_dynamics = PBFPowerFieldCylindrical(nzs[i], nr, Δr, Δz, k[i], ρ[i], cₚ[i], T∞, h, Ty(step_time), Ty, V, M; buffer=buffers[i], end_layer=true)
        else
            temp_dynamics = PBFPowerFieldCylindrical(nzs[i], nr, Δr, Δz, k[i], ρ[i], cₚ[i], T∞, h, Ty(step_time), Ty, V, M; buffer=buffers[i])
        end
        temp_tempering = Tempering(lnA, n, E, Ty(step_time); num=nr*nzs[i])
        dynamics[counter] = PBFTempering_cylindrical(temp_dynamics, temp_tempering)
        U[counter] = V([0; 0])
        dts[counter] = Ty(step_time)
        global counter += 1
        global h_test += step_time*h
    end

    # Post build cooling
    if i == nz
        temp_dynamics = PBFPowerFieldCylindrical(nzs[i], nr, Δr, Δz, k[i], ρ[i], cₚ[i], T∞, h, post_build_dt, Ty, V, M; buffer=buffers[i])
        temp_tempering = Tempering(lnA, n, E, post_build_dt; num=nr*nzs[i])
        for j = 1:post_process_steps
            dynamics[counter] = PBFTempering_cylindrical(temp_dynamics, temp_tempering)
            U[counter] = V([0; 0])
            dts[counter] = post_build_dt
            global counter += 1
            global h_test += post_build_dt*h
        end
    end

    update!(progress, effort[i])
end

println("Dynamics Initialized")

## Optimization Setup

println("Initialize Optimization")
process = Process(length(dynamics), dynamics, V)
Nu = length(U[1])

constraints = Vector{Constraint{Ty}}(undef, total_steps)
costs = Vector{Cost{Ty}}(undef, total_steps)

# Constraints
Pmin, Pmax = 0.200 * η, 0.350 * η # kW ## DEFAULT .200 to .350
Tbmin, Tbmax, TbRoC = 0.3, 1.1, 0.01/60*1e1 # kK, kK/s
# Tmax = 1.2 # kK (Should be based on total simulation time and the nose of the TTT curve) ## DEFAULT 1.1
y_target = 0.25
cutoff = 0.005 # m to ignore on bottom of inverted pyramid
counter = 1 # Keep track of the current time step within the optimization construction

top_temps = load_object("$example_path/52_deg_TopTemps_nz44.jld2")
top_setpoint = 10.5 # Temperature above which the maximum power constraint will begin to decrease


effort = cumsum(map(i -> (nzs[i] * nr)^3, 1:nz))
progress = Progress(effort[end], desc="Preparing layer dynamics matrices...")

for i in 1:nz
    nvox = nr * nzs[i]

    # Target state and input
    temp_x̄_c = V([T∞ * ones(nvox); 0; T_base; log.(-log.(ones(nvox) .* (1 - y_target)))])
    temp_x̄_p = V([T∞ * ones(nvox); P_set; T_base; log.(-log.(ones(nvox) .* (1 - y_target)))])
    temp_ū = V(zeros(Nu))
    temp_ū_p = V([P_set; 0])

    # Matrices used in quadratic cost
    Q = diagm([zeros(nvox) .+ 0.0; 0.00; 0.00; 1e-6 / nvox * ones(nvox) .* vec(layer_booleans[i]) .+ 0.00])
    # Q = diagm(ones(2nvox+2))
    R_opt = diagm(zeros(Nu))
    # R_opt = diagm(ones(Nu)*1e-5)
    Q = M(Q)
    R_opt = M(R_opt)

    ### No constraints for testing purposes only. ###
    A_x_eq = M(undef, 0, 2nvox+Nu)
    b_x_eq = V(undef, 0)

    A_u_eq = M(undef, 0, Nu)
    b_u_eq = V(undef, 0)

    A_x_ineq = M(undef, 0, 2nvox+Nu)
    b_x_ineq = V(undef, 0)

    A_u_ineq = M(undef, 0, Nu)
    b_u_ineq = V(undef, 0)

    BlankConstraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)
    ### End no constraints ###

    ## Add Layer ##
    if i > 1
        A_x_eq = M(begin
            m = zeros(1, 2nvox+Nu);
            m[1, nvox+1] = 1;
            m
        end) # Constrain power to 0
        b_x_eq = V([0])

        A_u_eq = M(undef, 0, Nu) # No equality constraints on inputs
        b_u_eq = V(undef, 0)

        A_x_ineq = M(begin
            m = zeros(1, 2nvox+Nu);
            m[1, nvox+2] = 1;
            [m; -m]
        end) # Constrain baseplate temperature between Tbmin and Tbmax
        b_x_ineq = V([Tbmax; -Tbmin])

        A_u_ineq = M([[0 1]; [0 -1]]) # Constrain baseplate temperature change between +TbRoC and -TbRoC
        b_u_ineq = V([TbRoC; TbRoC])

        costs[counter] = QuadraticCost(Q, R_opt, temp_x̄_c, temp_ū_p)
        constraints[counter] = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

        # constraints[counter] = BlankConstraint

        global counter += 1
    end

    ## Applied Power ##
    A_x_eq = M(undef, 0, 2(nvox)+Nu) # No equality constraints on states
    b_x_eq = V(undef, 0)

    A_u_eq = M([1 0]) # Constrain power change to 0
    b_u_eq = V([0])

    A_x_ineq = M(begin
        m1 = zeros(1, 2nvox+Nu);
        m1[1, nvox+1] = 1;
        m2 = zeros(1, 2nvox+Nu);
        m2[1, nvox+2] = 1;
        [m1; m2; -m1; -m2]
    end) # Constrain baseplate temperature between Tbmin and Tbmax and Power between Pmin and Pmax
    # b_x_ineq = V([Pmax; Tbmax; -Pmin; -Tbmin])

    P_shift = (top_temps[i] > top_setpoint) * (top_temps[i] - top_setpoint) * 0.25 * η
    b_x_ineq = V([(Pmax - P_shift); Tbmax; -Pmin; -Tbmin])


    A_u_ineq = M([[0 1]; [0 -1]]) # Constrain baseplate temperature change between +TbRoC and -TbRoC
    b_u_ineq = V([TbRoC; TbRoC])

    powered_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

    for k in 1:length(powered_steps[i])
        if k == length(powered_steps[i]) # The last value here has to unconstrain power change
            A_u_eq = M(undef, 0, Nu) # No equality constraints on inputs
            b_u_eq = V(undef, 0)

            costs[counter] = QuadraticCost(Q, R_opt, temp_x̄_p, -temp_ū_p)
            constraints[counter] = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

            # constraints[counter] = BlankConstraint
        else
            costs[counter] = QuadraticCost(Q, R_opt, temp_x̄_p, temp_ū)
            constraints[counter] = powered_constraint

            # constraints[counter] = BlankConstraint
        end

        global counter += 1
    end

    ## No Applied Power ##
    # Here we need: equality constraint on power = 0, inequality constraint on baseplate temperatre, inequality constraint on baseplate temperature ROC, no constraint on power change input
    A_x_eq = M(begin
        m = zeros(1, 2nvox+Nu);
        m[1, nvox+1] = 1;
        m
    end) # Constrain power to 0
    b_x_eq = V([0])

    A_u_eq = M(undef, 0, Nu) # No equality constraints on inputs
    b_u_eq = V(undef, 0)

    A_x_ineq = M(begin
        m = zeros(1, 2nvox+Nu);
        m[1, nvox+2] = 1;
        [m; -m]
    end) # Constrain baseplate temperature between Tbmin and Tbmax
    b_x_ineq = V([Tbmax; -Tbmin])

    A_u_ineq = M([[0 1]; [0 -1]]) # Constrain baseplate temperature change between +TbRoC and -TbRoC
    b_u_ineq = V([TbRoC; TbRoC])

    cooling_constraint = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

    for _ in 1:length(cooling_steps[i])
        costs[counter] = QuadraticCost(Q, R_opt, temp_x̄_c, temp_ū)
        constraints[counter] = cooling_constraint

        # constraints[counter] = BlankConstraint

        global counter += 1
    end

    # Post build cooling
    # Here we need: equality constraint on power = 0, clamping inequality constraint on baseplate temperatre, inequality constraint on baseplate temperature ROC, no constraint on power change input
    if i == nz
        A_x_eq = M(begin
            m = zeros(1, 2nvox+Nu);
            m[1, nvox+1] = 1;
            m
        end) # Constrain power to 0
        b_x_eq = V([0])

        A_u_eq = M(undef, 0, Nu) # No equality constraints on inputs
        b_u_eq = V(undef, 0)

        A_x_ineq = M(begin
            m = zeros(1, 2nvox+Nu);
            m[1, nvox+2] = 1;
            [m; -m]
        end) # Constrain baseplate temperature between Tbmin and Tbmax
        b_x_ineq = V([Tbmax; -Tbmin])

        A_u_ineq = M([[0 1]; [0 -1]]) # Constrain baseplate temperature change between +TbRoC and -TbRoC
        b_u_ineq = V([TbRoC; TbRoC])

        for j = 1:post_process_steps
            # b_x_ineq = V([max(Tbmin+.01, Tbmax - post_build_dt*TbRoC*j); -Tbmin])

            if j == post_process_steps
                b_x_ineq = V([Tbmin+0.01; -Tbmin])
            end


            constraints[counter] = LinearConstraint(A_x_eq, b_x_eq, A_u_eq, b_u_eq, A_x_ineq, b_x_ineq, A_u_ineq, b_u_ineq)

            # constraints[counter] = BlankConstraint

            costs[counter] = QuadraticCost(Q, R_opt, temp_x̄_c, temp_ū)

            if j >= post_process_steps # Final state costs
                temp_x̄_f = V([T∞ * ones(nvox); 0; Tbmin; log.(-log.(ones(nvox) .* (1 - y_target)))])
                final_cost_booleans = Float32.(layer_booleans[i])
                final_cost_booleans[(end-Int(floor(cutoff/Δz))):end, :] .= 0
                # final_cost_booleans[1:10, :] .*= 0.001 # Set target hardness lower at the top

                Qf = diagm([zeros(nvox) .+ 0.00; 0.00; 0.00; 1e-2 / nvox * ones(nvox) .* vec(final_cost_booleans) .+ 0.00])
                # Qf = diagm(ones(2nvox+2))
                # Rf = diagm(zeros(Nu))
                Rf = diagm(ones(Nu)*1e-4)
                Qf = M(Qf)
                Rf = M(Rf)
                costs[counter] = QuadraticCost(Qf, Rf, temp_x̄_f, temp_ū)
            end
            global counter += 1
        end
    end

    update!(progress, effort[i])
end

println("Optimization Initialized")

problem = Problem(x₀, process, costs, constraints, M)

rollout!(problem, U)
rollout!(problem, U)

# Initial Statistics
ŷ_surface_default = reverse(reshape(Array(problem.z.X[end][(Int(length(problem.z.X[end])÷2)+2):end]), Int(length(problem.z.X[end])÷nr÷2), nr), dims=1)
y_surface_default = 1 .- exp.(-exp.(ŷ_surface_default))
default_hardness = sqrt.(((Array(y_surface_default .* reverse(boolean_mesh, dims=1)) .- 0.001000000370471716) .* (460^2 - 290^2) ./ (0.3365735558787163 - 0.001000000370471716)) .+ 290^2) .+ (320 - 290)

eval_constraints!(problem, problem.z, problem.v)
eval_penalty_multiplier!(problem, problem.v, 1e-1)
@show constraint_violation(problem, problem.v)
eval_lagrangian_cost(problem.v)

@show eval_cost(problem)
@time al_ddp!(problem; ctol=3e-5, μ=0.1e3, ϕ=3.0, verbosity=3, ρi=1e-10, tol=1e-7, gtol=P_set / Nu / 100, maxiters=35) # calling the optimization #tol was 1e-6
# @time al_ddp!(problem; ctol=1e-4, μ=0.1, ϕ=3.0, verbosity=3, ρi=1e-10, tol=1e-4, gtol=P_set / Nu / 100) # You have to do it twice for some reason
@show eval_cost(problem)

U_opt = [Array(u) for u in problem.z.U] #extract the problem structure's stored inputs, which are now solved. Array converts back to CPU vectors
X_opt = [Array(x) for x in problem.z.X] #extract the problem structure's stored inputs, which are now solved. Array converts back to CPU vectors

# Saving inputs and states
# save_object("U_optimized_$(y_target)_target.jld2", U_opt)
# save_object("X_optimized_$(y_target)_target.jld2", X_opt)

## Smoothing power input to what the machine would actually run ##
# powered_locs = Array(reduce(hcat, U)'[:, 1]) .> 0.05
p_start = 1
p_end = 1

U_smooth = zeros(total_steps, 2) #Vector{}(undef, total_steps)

for i in 2:(total_steps-1)
    if powered_locs[i] > powered_locs[i-1]
        global p_start = i

        temp_time_sum = sum(dts[(p_end+1):(p_start-1)])

        p_smooth = sum([X_opt[j][length(X_opt[j])÷2] * dts[j] / temp_time_sum for j in (p_end+1):(p_start-1)])

        U_smooth[(p_end+1):(p_start-1), :] .= [p_smooth 0]
    end
    if powered_locs[i+1] < powered_locs[i]
        global p_end = i

        temp_time_sum = sum(dts[p_start:p_end])

        p_smooth = sum([X_opt[j][length(X_opt[j])÷2] * dts[j] / temp_time_sum for j in p_start:p_end])

        U_smooth[p_start:p_end, :] .= [p_smooth 0]
    end
end

## Create smoothed baseplate temp inputs for testing
setpoints = [
    0 423;
    0.8 890;
    1.2 760;
    1.5 800
    1.9 640;
    2.2 640;
    3.0 900;
    5.3 1025;
    6.9 300
]

setpoints_for_input = copy(setpoints)
setpoints_for_input[:, 1] .*= 3600
setpoints_for_input[:, 2] .*= 1e-3
rates = setpoints_for_input[:, 2] ./ setpoints_for_input[:, 1]
rates = [(setpoints_for_input[i+1, 2] - setpoints_for_input[i, 2]) / (setpoints_for_input[i+1, 1] - setpoints_for_input[i, 1]) for i in 1:(size(setpoints_for_input, 1)-1)]

# Time steps for both visualizations
skip_steps = 1
t = (1:skip_steps:total_steps) # Time vector
dts_summed = [sum(dts[1:i]) for i in 1:skip_steps:total_steps]

T_smooth_0 = zeros(length(problem.z.X))
T_smooth_0[1] = 0#setpoints_for_input[:, 1]
T_smooth_0[end] = 0
for i in 2:(length(T_smooth_0)-1)
    current_time = dts_summed[i]
    current_setpoint = Int(findfirst(x -> x > current_time, setpoints_for_input[:, 1]))-1
    T_smooth_0[i] = rates[current_setpoint]
    # T = rates[current_setpoint]
end

# Smoothed inputs with laser power smoothing only
T_smooth = [Array(x)[length(x)÷2+1] for x in problem.z.X]
U_smooth[1, 1] = Array(problem.z.X[1][(length(problem.z.X[1])÷2):(length(problem.z.X[1])÷2)])[1]
U_smooth[:, 2] .= T_smooth
U_smooth[abs.(U_smooth) .< 0.01] .= 0

U_smooth_inputs = zeros(size(U_smooth))
U_smooth_inputs[1:(end-1), 1] = [U_smooth[i+1, 1] - U_smooth[i, 1] for i in 1:(total_steps-1)]
U_smooth_inputs[1:(end-1), 2] = [U_smooth[i+1, 2] - U_smooth[i, 2] for i in 1:(total_steps-1)]
U_smooth_inputs ./= dts
U_smooth_inputs[:, 1] .*= 1e-1
U_smooth_inputs[:, 2] .*= 1e1
U_smooth_inputs = [V(U_smooth_inputs[i, :]) for i in 1:total_steps]

# Smoothed inputs with laser power and baseplate temperature smoothing
U_smooth_inputs_2 = zeros(size(U_smooth))
U_smooth_inputs_2[1:(end-1), 1] = [U_smooth[i+1, 1] - U_smooth[i, 1] for i in 1:(total_steps-1)]
U_smooth_inputs_2[1:(end-1), 2] = T_smooth_0[2:end]
U_smooth_inputs_2[:, 1] ./= dts
U_smooth_inputs_2[:, 1] .*= 1e-1
U_smooth_inputs_2[:, 2] .*= 1e1
U_smooth_inputs_2 = [V(U_smooth_inputs_2[i, :]) for i in 1:total_steps]

## End Smoothing ##

# Time steps for both visualizations
skip_steps = 1
t = (1:skip_steps:total_steps) # Time vector
dts_summed = [sum(dts[1:i]) for i in 1:skip_steps:total_steps]

### VISUALIZE OPTIMAL PROBLEM ###
T_field_optimal = [reverse(reshape(Array(x[1:Int((length(x)-Nu)÷2)] .* 1e3), Int((length(x) - Nu)÷nr÷2), nr), dims=1) for x in problem.z.X[1:skip_steps:end]] # Temperature (K) from state variables
P_app_optimal = [Array(x[(length(x)÷2):(length(x)÷2+1)]) for x in problem.z.X[1:skip_steps:end]] # Power and baseplate temperature from state variables 
P_app_plot_optimal = [Array(reduce(hcat, P_app_optimal[1:i])') for i in 1:skip_steps:length(problem.z.U)]
ŷ_surface_optimal = [reverse(reshape(Array(x[(Int(length(x)÷2)+2):end]), Int(length(x)÷nr÷2), nr), dims=1) for x in problem.z.X[1:skip_steps:end]] # Transformation amount from state variables
y_surface_optimal = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface_optimal] # Transforming back to transformation percentages 

# Plot Parameters
default(fontfamily="Helvetica")

# anim = @animate for (t, P, T, y) in zip(t[1:end], P_app_plot_optimal[1:end], T_field_optimal[1:end], y_surface_optimal[1:end])

#     title_str = @sprintf "Process at %.3f seconds" dts_summed[t]

#     h1 = scatter(dts_summed[1:skip_steps:t], P[:, 1] ./ η, xlabel="Time (s)", ylabel="Power (W)", label = missing, markerstrokewidth=0, marker=:square) # , label="Applied Power"
#     scatter!(h1, dts_summed[1:skip_steps:t], P[:, 2], ylabel="Temperature (K)", label = missing, markerstrokewidth=0, marker=:circle) # , label="Baseplate Temperature", yaxis=:right
#     hline!(h1, [Pmax / η], linestyle=:dash, color=:blue, label = missing)
#     hline!(h1, [Pmin / η], linestyle=:dash, color=:blue, label = missing)
#     h2 = heatmap(T, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(0.275 * 1e3, 1.1 * 1e3), title=title_str, aspect_ratio=:equal) # Temperature
#     h3 = heatmap(y, ticks=false, c=:acton, cbar_title="Fraction Transformed", clim=(0, 1), aspect_ratio=:equal) # Tempering

#     plot(h1, h2, h3; layout=(1, 3), size=(1150, 600)) 
#     # plot(h2, h3; layout=(1, 3), size=(1150, 600)) 

# end
# gif(anim, "ConstantPower_optimized_target_$(y_target)_$(n_subsample)_optimal.mp4", fps=(1 / skip_steps * 3))
# gif(anim, "NaiveExample_$(n_subsample)_optimal.mp4", fps=(5 / skip_steps))
### END OPTIMAL VISUALIZATION ###

### VISUALIZE SMOOTHED PROBLEM ###
# rollout!(problem, U_smooth_inputs) # Rollout smoothed problem (laser power only)
rollout!(problem, U_smooth_inputs_2) # Rollout smoothed problem (laser power and baseplate temp)

T_field_smoothed = [reverse(reshape(Array(x[1:Int((length(x)-Nu)÷2)] .* 1e3), Int((length(x) - Nu)÷nr÷2), nr), dims=1) for x in problem.z.X[1:skip_steps:end]] # Temperature (K) from state variables
P_app_smoothed = [Array(x[(length(x)÷2):(length(x)÷2+1)]) for x in problem.z.X[1:skip_steps:end]] # Power and baseplate temperature from state variables 
P_app_plot_smoothed = [Array(reduce(hcat, P_app_smoothed[1:i])') for i in 1:skip_steps:length(problem.z.U)]
ŷ_surface_smoothed = [reverse(reshape(Array(x[(Int(length(x)÷2)+2):end]), Int(length(x)÷nr÷2), nr), dims=1) for x in problem.z.X[1:skip_steps:end]] # Transformation amount from state variables
y_surface_smoothed = [1 .- exp.(-exp.(ŷ)) for ŷ in ŷ_surface_smoothed] # Transforming back to transformation percentages 

# anim = @animate for (t, P, T, y) in zip(t[1:end], P_app_plot_smoothed[1:end], T_field_smoothed[1:end], y_surface_smoothed[1:end])

#     title_str = @sprintf "Process at %.3f seconds" dts_summed[t]

#     h1 = scatter(dts_summed[1:skip_steps:t], P[:, 1] ./ η, xlabel="Time (s)", ylabel="Power (W)", label = missing, markerstrokewidth=0, marker=:square) # , label="Applied Power"
#     scatter!(h1, dts_summed[1:skip_steps:t], P[:, 2], ylabel="Temperature (K)", label = missing, markerstrokewidth=0, marker=:circle) # , label="Baseplate Temperature", yaxis=:right
#     hline!(h1, [Pmax / η], linestyle=:dash, color=:blue, label = missing)
#     hline!(h1, [Pmin / η], linestyle=:dash, color=:blue, label = missing)
#     h2 = heatmap(T, ticks=false, c=:inferno, cbar_title="Temperature (K)", clim=(0.275 * 1e3, 1.1 * 1e3), title=title_str, aspect_ratio=:equal) # Temperature
#     h3 = heatmap(y, ticks=false, c=:acton, cbar_title="Fraction Transformed", clim=(0, 1), aspect_ratio=:equal) # Tempering

#     plot(h1, h2, h3; layout=(1, 3), size=(1150, 600)) 

# end
# gif(anim, "ConstantPower_optimized_target_$(y_target)_$(n_subsample)_smoothed.mp4", fps=(1 / skip_steps * 3))
### END SMOOTHED VISUALIZATION ###

# plot([maximum(x) for x in T_field_optimal])

# scatter(P_app_optimal[end][:, 1] / η)

# scatter(Array(reshape(y_surface_smoothed[end], nz, nr)[:, 1]), xlabel = "Position (mm)", ylabel = "Fraction Transformed")

## Controlled Input Plot
U_smooth[U_smooth .== 0] .= NaN
axis1 = plot(dts_summed ./ 3600, U_smooth[:, 1] ./ η .* 1e3, linewidth=3, label="Optimal Power", color="#1f77b4")
scatter!(dts_summed[1:30] ./ 3600, U_smooth[1:30, 1] ./ η .* 1e3, linewidth=3, label=missing, color="#1f77b4", markersize=1.6, markerstrokewidth=0)
plot!(dts_summed ./ 3600, U_smooth[:, 2] .* 1e3, linewidth=2, label="Optimal Baseplate Temperature", color="#ff7f0e")
# axis1 = scatter(dts_summed[U_smooth[:, 1] .> 0] ./3600, U_smooth[:, 1][U_smooth[:, 1] .> 0] ./ η  .* 1e3, label = "Applied Power")
# scatter!(dts_summed ./ 3600, U_smooth[:, 2] .* 1e3, label = "Baseplate Temperature")
axis2 = twinx()
scatter!(axis2, ylims=ylims(axis1))
hline!(axis1, [285], color="#1f77b4", linestyle=:dash, label="Default Power")
hline!(axis1, [423], color="#ff7f0e", linestyle=:dash, label="Default Baseplate Temperature")
xlabel!(axis1, "Time (Hours)")
xlabel!(axis2, "")
ylabel!("Power (W)")
ylabel!(axis2, "Baseplate Temperature (K)")
title!("Controlled Inputs")

## Controlled Input Plot Different Axes No Hlines
U_smooth[U_smooth .== 0] .= NaN
axis1 = plot([NaN], [NaN], linewidth=2*1.5, label="Optimal Baseplate Temperature", color="#ff7f0e", dpi=500)
axis2 = twinx()
plot!(axis2, dts_summed ./ 3600, U_smooth[:, 2] .* 1e3, linewidth=2*1.5, label=missing, color="#ff7f0e")
plot!(dts_summed ./ 3600, U_smooth[:, 1] ./ η .* 1e3, linewidth=3*1.5, label="Applied Power", color="#1f77b4")
scatter!(dts_summed[1:45] ./ 3600, U_smooth[1:45, 1] ./ η .* 1e3, label=missing, color="#1f77b4", markersize=1.6*1.5, markerstrokewidth=0)
xlabel!(axis1, "Time (Hours)")
xlabel!(axis2, "")
ylabel!(axis1, "Power (W)")
ylabel!(axis2, "Baseplate Temperature (K)")
title!(axis1, "Controlled Inputs")

setpoints = [
    0 423;
    0.8 890;
    1.2 760;
    1.5 800
    1.9 640;
    2.2 640;
    3.0 900;
    5.3 1025;
    6.9 300
]

setpoints[:, 2] .-= 273f0

plot!(axis1, [NaN], [NaN], label="Actual Baseplate Temperature", color=:black, linewidth=1.25)
plot!(axis2, setpoints[:, 1], setpoints[:, 2], label=missing, color=:black, linewidth=1.25)

# Baseplate temperature setpoints to use to run build
# setpoints = [ # for 0.01 IC 10s dwell
#     0 423;
#     0.65 700;
#     1.05 700;
#     1.65 455;
#     2.2 455;
#     2.75 375;
#     3.4 375;
#     5.75 555;
#     6.5 930; 
#     8.06 300
# ]

# setpoints = [ # for 0.001 IC 10s dwell
#     0 423;
#     0.78 840;
#     1.65 480;
#     2.27 480;
#     2.8 310;
#     3.7 305;
#     4.6 470;
#     5.2 505;
#     5.62 480;
#     6.48 935; 
#     8.06 300
# ]

# savefig("ControlledInputs.pdf")

## Uncontrolled Input Plot
naive_power = ones(total_steps) .* P_set ./ η .* 1e3
naive_power[isnan.(U_smooth[:, 1])] .= NaN
naive_baseplate = ones(total_steps) .* T_base * 1e3
naive_baseplate[dts_summed .> dts_summed[end]-post_process_time] .= T∞ * 1e3
axis3 = plot(dts_summed ./ 3600, naive_power[:, 1], label="Applied Power", ylims=ylims(axis1), linewidth=3, color="#1f77b4")
scatter!(dts_summed[1:44] ./ 3600, naive_power[1:44], linewidth=3, label=missing, color="#1f77b4", markersize=1.6, markerstrokewidth=0)
plot!(dts_summed ./ 3600, naive_baseplate, label="Baseplate Temperature", color="#ff7f0e", linewidth=2)
axis2 = twinx()
plot!(axis2, ylims=ylims(axis1))
xlabel!(axis3, "Time (Hours)")
xlabel!(axis2, "")
ylabel!(axis3, "Power (W)")
ylabel!(axis2, "Baseplate Temperature (K)")
title!("Uncontrolled Inputs")

## Final Transformation Percentage Plot with and without powder
final_hardness_optimal = sqrt.(((Array(reshape(y_surface_optimal[end], nz, nr) .* reverse(boolean_mesh, dims=1)) .- 0.001000000370471716) .* (460^2 - 290^2) ./ (0.3365735558787163 - 0.001000000370471716)) .+ 290^2) .+ (320 - 290)
final_hardness_smoothed = sqrt.(((Array(reshape(y_surface_smoothed[end], nz, nr) .* reverse(boolean_mesh, dims=1)) .- 0.001000000370471716) .* (460^2 - 290^2) ./ (0.3365735558787163 - 0.001000000370471716)) .+ 290^2) .+ (320 - 290)
final_hardness_smoothed = final_hardness_smoothed .* reverse(boolean_mesh, dims=1)
final_hardness_smoothed[final_hardness_smoothed .== 0] .= NaN
default_hardness = default_hardness .* reverse(boolean_mesh, dims=1)
default_hardness[default_hardness .== 0] .= NaN
heatmap(final_hardness_optimal, ticks=false, c=:plasma, cbar_title="Hardness (HV)", clim=(290, 550), aspect_ratio=:equal, title="Optimal")#, axis=false)
heatmap(final_hardness_smoothed[19:end, 1:17], ticks=false, c=:plasma, cbar_title="Hardness (HV)", clim=(290, 550), aspect_ratio=:equal, title="Smoothed")#, axis=false)
# savefig("FinalHardnessDistribution.pdf")
heatmap(default_hardness[19:end, :], ticks=false, c=:plasma, cbar_title="Hardness (HV)", clim=(290, 550), aspect_ratio=:equal, title="Smoothed")#, axis=false)
# savefig("DefaultHardnessDistribution_cutoff.pdf")
scatter((1:nz)Δz*1000, final_hardness_smoothed[:, 1])
scatter(final_hardness_optimal[:, 1])

target_hardness = sqrt.(((y_target .- 0.001000000370471716) .* (460^2 - 290^2) ./ (0.3365735558787163 - 0.001000000370471716)) .+ 290^2) .+ (320 - 290)

boolean_mesh_considered = copy(boolean_mesh)
boolean_mesh_considered[21:end, :] .*= 0
boolean_mesh_considered[1:1, :] .*= 0
boolean_mesh_considered[:, 18:end] .*= 0 # Uncomment to ignore outer overhang region in statistics calculations


default_average_hardness = sum(default_hardness[reverse(boolean_mesh_considered .== 1, dims=1)]) / count(boolean_mesh_considered .== 1)
default_hardness_stdev = sqrt(sum((default_hardness[reverse(boolean_mesh_considered .== 1, dims=1)] .- default_average_hardness) .^ 2)/count(boolean_mesh_considered .== 1))

opt_average_hardness = sum(final_hardness_smoothed[reverse(boolean_mesh_considered .== 1, dims=1)]) / count(boolean_mesh_considered .== 1)
opt_hardness_stdev = sqrt(sum((final_hardness_smoothed[reverse(boolean_mesh_considered .== 1, dims=1)] .- opt_average_hardness) .^ 2)/count(boolean_mesh_considered .== 1))

# Laser power input for each layer
final_power_inputs = [isnan(U_smooth[i+1, 1]) && !isnan(U_smooth[i, 1]) ? U_smooth[i, 1] / η * 1e3 : missing for i in 1:(total_steps-1)]
final_power_inputs = Float32.(final_power_inputs[.!ismissing.(final_power_inputs)])

@show default_average_hardness
@show default_hardness_stdev

@show target_hardness
@show opt_average_hardness
@show opt_hardness_stdev
# @show h_avg = h_test / total_time_hrs / 3600

# save_object("UncertaintyData/Final_Hardness_E-3s.jld2", final_hardness_smoothed)

trace_heights = (1:nz)*55/nz*1e-3 #(5:5:55) * 1e-3 #m 
trace_locs = Int.(round.(trace_heights / Δz))
traces = Vector{}(undef, length(trace_heights))

for (j, loc) in enumerate(trace_locs)
    traces[j] = [round(loc*Δz*1e3, digits=1) NaN]
    for i in 1:total_steps
        if size(T_field_optimal[i], 1) >= loc
            traces[j] = [traces[j]; [dts_summed[i] T_field_optimal[i][loc, 1] - 273]]
        end
    end
end

test_trace = 6
test = traces[test_trace][2:end, :]

plot(test[:, 1], test[:, 2], label=missing)
xlabel!("Time (s)")
ylabel!("Temperature (C)")
title!("Temperature Trace $(traces[test_trace][1, 1])mm from bottom")

# save_object("Temperatre_Traces_Updated_again_final.jld2", traces)

scatter(default_hardness[:, 1])
scatter!(final_hardness_smoothed[:, 1])

# save_object("SimulatedOptimalHardness.jld2", final_hardness_smoothed[:, 1])
# save_object("SimulatedDefaultHardness.jld2", default_hardness[:, 1])