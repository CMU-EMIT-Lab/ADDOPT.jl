# EB-PBF Minimum Thermal Variance
To run this example, either run `ebpbf_variance_example.jl` from REPL or the CLI. This will generate animation and CSV spot pattern files in the working directory.
```
julia --project examples/ebpbf/variance/ebpbf_variance_example.jl
```

The CSVs can be converted to OBP files for use with the Freemelt ONE using `ebamareaprint.py` in the parent directory. Please reference the corresponding README.

This study has been most recently tested on Ubuntu 24.04 x86_64, with a NVIDIA RTX 4090 GPU, Julia version 1.12.6, Python version 3.12.3. Total runtime was slightly under one hour.