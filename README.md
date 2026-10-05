# ADDOPT.jl

An AL-DDP based solver for ADDitive manufacturing with OPTimal trajectories.

## Examples
Example studies using the library can be found in the `examples` folder. 

- `examples/furnace_example.jl` Demonstrates the optimization method on CPU with a simple "furnace" dynamics for controlling transformation.
- `examples/ebpbf` Contains two problems for electron beam powder bed fusion and accompanying conversion python scripts.
- `examples/ebpbf/variance` Solves the minimum thermal variance problem on GPU.
- `examples/ebpbf/hardness` Solves the prescribed hardness control problem for EB-PBF of L-59 steel on GPU.
- `examples/lpbf/in718` Solves the prescribed hardness control problem for L-PBF of IN718 on GPU.

## Published Work

The minimal thermal variance work has been published. If you use or reference it, we kindly request you cite the [following paper](https://doi.org/10.1115/1.4067325):
```bibtex
@article{KhrenovASME2025,
    author = {Khrenov, Mikhail and Frieden Templeton, William and Prabha Narra, Sneha},
    title = {{ADDOPT}: An Additive Manufacturing Optimal Control Framework Demonstrated in Minimizing Layer-Level Thermal Variance in Electron Beam Powder Bed Fusion},
    journal = {Journal of Manufacturing Science and Engineering},
    volume = {147},
    number = {4},
    pages = {041009},
    year = {2025},
    month = 1,
    issn = {1087-1357},
    doi = {10.1115/1.4067325},
}
```

The EB-PBF steel hardness control work has been published. If you use or reference it, we kindly request you cite the [following proceedings paper](https://doi.org/10.23919/ACC63710.2025.11107816):
```bibtex
@INPROCEEDINGS{KhrenovACC2025,
    author={Khrenov, Mikhail and Tan, Moon and Fitzwater, Lauren and Hobdari, Michelle and Narra, Sneha Prabha},
    booktitle={2025 American Control Conference (ACC)}, 
    title={Trajectory Optimization for Spatial Microstructure Control in Electron Beam Metal Additive Manufacturing}, 
    year={2025},
    month=7,
    volume={},
    number={},
    pages={541-546},
    doi={10.23919/ACC63710.2025.11107816}
}
```

The L-PBF IN718 hardness control work has been published. If you use or reference it, we kindly request you cite the [following paper](https://doi.org/10.1016/j.addma.2026.105205):
```bibtex
@article{FriedenMcCauleyAM2026,
    title = {Property optimization through full-part thermal history control in laser powder bed fusion additive manufacturing},
    journal = {Additive Manufacturing},
    volume = {123},
    pages = {105205},
    year = {2026},
    issn = {2214-8604},
    doi = {10.1016/j.addma.2026.105205},
    author = {William J. {Frieden Templeton} and Jacob A. McCauley and Mikhail Khrenov and Shawn Hinnebusch and Miguel Pena and Lin Shao and Albert C. To and Sneha Prabha Narra},
}
```