function boolean_vector(boolean_mesh, solid_property, reduction_factor)
    # Generate a vector for a certain property that changes based on a boolean matrix input
    # Requires input of the boolean mesh, solid property, and the reduction factor for the powder
    # Set recution factor to 1 for properties that are the same for powder and metal (z heights, for example)

    # Generate initial vector framework
    vector = ones(Float32, size(boolean_mesh))

    # Set all values to powder values, then set solid sections to solid values
    vector = vector*solid_property*reduction_factor
    vector = (vector .* boolean_mesh ./ reduction_factor) .+ (vector .* (1 .- boolean_mesh))

    # Convert to CUDA vector
    vector = vec(vector)

    return vector
end 

function boolean_vector(boolean_mesh, solid_property, reduction_factor, type)
    # Generate a vector for a certain property that changes based on a boolean matrix input
    # Requires input of the boolean mesh, solid property, and the reduction factor for the powder
    # Set recution factor to 1 for properties that are the same for powder and metal (z heights, for example)

    # Generate initial vector framework
    vector::type = ones(Float32, size(boolean_mesh))

    # Set all values to powder values, then set solid sections to solid values
    vector = vector*solid_property*reduction_factor
    vector = (vector .* boolean_mesh ./ reduction_factor) .+ (vector .* (1 .- boolean_mesh))

    # Convert to vector
    vector = vec(vector)

    return vector
end 

function find_boundary_voxels(input_mesh)
    # Takes the input of a boolean mesh
    # returns a corresponding boolean matrix with a one at every voxel that is at the border between a 1 and a 0
    output = zeros(size(input_mesh))
    nz, nr = size(input_mesh)
    for i in 1:nz
        for j in 1:nr
            if (i > 1 && input_mesh[i-1, j] != input_mesh[i, j]) || (i < nz && input_mesh[i+1, j] != input_mesh[i, j]) ||
                (j > 1 && input_mesh[i, j-1] != input_mesh[i, j]) || (j < nr && input_mesh[i, j+1] != input_mesh[i, j])
                 output[i, j] = 1
             end
        end
    end
    return output
end

function to_coarsen_layers(vector, dims, condition)
    # Takes a vector input, reshapes into mesh
    # Finds pairs of layers that correspond to a specified conditional statement 
    # Returns indexes of the top of each pair of repeated vectors that supports the conditional

    mesh = reshape(vector, (dims))

    coarsened_layers = Int[]

    for i in 1:1:(length(mesh[1, 1, :]) - 1) # Run through each layer
        if isequal(mesh[:, :, i], mesh[:, :, i+1]) &&  condition(mesh[:, :, i], mesh[:, :, i+1]) # Check for similarity between current and next layer and apply selected condition to these layers
            if isempty(coarsened_layers) || i - 1 != coarsened_layers[end] # Make sure we don't flag two consecutive layers, second half is coarsening condition
                push!(coarsened_layers, i)
            end    
        end
    end

    return coarsened_layers
end

function coarsen_vector(vector, dims, layers, scale_factor)
    # input a vector, its dimensions, the layers to coarsen, and a scale factor 
    # multiply the specified layers by (either 2 for z heights or 1 for intrinsic variables)
    # remove layer BELOW the specified layers, aka higher index value

    reshaped_vec = reshape(vector, (dims))

    # Multiply selected layers by scale factor, remove subsequent layers
    reshaped_vec[:, :, layers] = reshaped_vec[:, :, layers] .* scale_factor
    reshaped_vec = reshaped_vec[:, :, setdiff(1:end, layers .+ 1)]

    # Calculate new mesh z dimension
    new_nz = length(reshaped_vec[1, 1, :])

    # Revert to vector format
    coarsened_mesh = vec(reshaped_vec)

    return coarsened_mesh, new_nz
end

function vector_to_disc(nr, x)
    # Takes one of the layers in the axisymmetric simulation, in the form of a vector with length < nz 
    # Returns a 2D matrix with the values from the layer in a disc
    c_matrix = zeros(2nr, 2nr)
    buffer = nr - length(x)
    for i = 1:2nr
        for j = 1:2nz
            if ((i-nr)^2 + (j-nr)^2)^0.5 < nr - buffer
                index = Int(ceil(((i-nr)^2 + (j-nr)^2)^0.5))
                    if index == 0 
                        index = 1 # Just so that the middle of the circle gets filled
                    end
                c_matrix[i, j] = x[index]
            end
        end
    end
    return c_matrix
end

function vector_to_disc(nr, x, slice)
    # Takes one of the layers in the axisymmetric simulation, in the form of a vector with length < nr 
    # Returns a 2D matrix with the values from the layer in a disc
    # If slice == true a slice is cut out of the discs
    c_matrix = fill(NaN, 2nr, 2nr)
    buffer = nr - length(x)
    for i = 1:2nr
        for j = 1:2nr
            if ((i-nr)^2 + (j-nr)^2)^0.5 < nr - buffer && (slice && (i-nr > 0 || j-nr > 0))
                index = Int(ceil(((i-nr)^2 + (j-nr)^2)^0.5))
                    if index == 0 
                        index = 1 # Just so that the middle of the circle gets filled
                    end
                c_matrix[i, j] = x[index]
            end
        end
    end
    return c_matrix
end

function remove_internal_voxels(data)
    # Takes a 3D matrix input with a solid numerical value and sets the internal values to nans to optimize 3D plotting
    # For each data point checks if there is a nan value around it, which means it is a surface point, and sets that value to true so it will not be removed. 
    nx, ny, nz = size(data)
    voxels_to_remove = zeros(Bool, (nx, ny, nz))
    for k = 2:nz-1
        for i = 2:nx-1
            for j = 2:ny-1
                if !(isnan(data[i-1, j, k]) || isnan(data[i+1, j, k]) || isnan(data[i, j-1, k]) || isnan(data[i, j+1, k]) || isnan(data[i, j, k-1]) || isnan(data[i, j, k+1]))
                    voxels_to_remove[i, j, k] = true
                end
            end
        end
    end

    data[voxels_to_remove] .= NaN

    return data
end