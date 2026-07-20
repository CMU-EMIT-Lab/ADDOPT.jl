# IN718 L-PBF Hardness Control
To run this example, either run `variable_layer_time_optimization.jl` from REPL or the CLI.
```
julia --project examples/lpbf/in718/variable_layer_time_optimization.jl
```

There are several files required to run this study:
1. A black and white image of the axisymmetric domain, where black corresponds to solid metal and whire corresponds to metal powder, `invertedpyramid52.png` in this example. `binary_modeler.jl` has several utilties to be used in conjunction with these black and white images.
2. The maximum temperature of the domain over time in a default parameter build, during the steps where there is nonzero laser power, `52_deg_Powered_Max_Temperatures_nz_44.jld2` in this example.
3. The maximum temperature of the domain over time in a default parameter build, during the steps where there is zero laser power, `52_deg_Unowered_Max_Temperatures_nz_44.jld2` in this example.

Item 1 can be generated using any 2D sketching software, AutoCAD was used in this study. Item 1 is used to assign material properties, costs, and constraints to each region of the domain correctly. 

Items 2 and 3 can be generated using any layer-by-layer thermal simulation methods. The temperature histories used in this work were generated using a forward simulation in ADDOPT.jl. ADDOPT.jl forward simulations with the same syntax are also used to calculate inital guesses for optimization studies. Relatively fine time steps are recommended for maximum accuracy. These files are used to calculate the maximum allowable time step length given limits on allowable transformation within a time step. 

This study has been most recently tested on Ubuntu 24.04 x86_64, with a NVIDIA RTX 4090 GPU, Julia version 1.12.6, CUDA.jl v5.8.5. Total runtime was roughly 45 minutes.

More recent versions of CUDA.jl affect converged solutions slightly due to differing operation ordering and rounding. 
