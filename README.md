# nonKam.jl

Julia tools for simulations of the kicked harmonic oscillator and their
quantum Fisher information (QFI).

## Supported layout

- `src/`: the `nonKam` package implementation. Public functions are exported
  from `src/nonKam.jl`.
- `scripts/`: maintained, runnable analyses and figure-generation scripts.
- `test/`: package tests. Run them with `julia --project=. -e 'using Pkg; Pkg.test()'`.
- `docs/`: package documentation source and generated local documentation.
- `data/`, `cache/`, and `plot/`: generated simulation data, caches, and
  figures. The `plot/` directory deliberately contains output files only.
- `legacy/`: preserved historical notebooks and scripts; not supported.

## Current entry points

| Goal | Use |
| --- | --- |
| QFI for one state | `floquet_qfi` |
| QFI over resonance values | `floquet_qfi_R_sweep` and `save_qfi_R_sweep` |
| Mean excitation number | `mean_n` |
| Excitation-number variance | `var_n` |
| QFI R sweep script | `scripts/qfi_time_R_sweep.jl` |
| Mean-energy figure | `scripts/plot_mean_energy_vs_time.jl` |
| Variance figure | `scripts/plot_var_n_vs_time.jl` |

For normal use, activate the project (`julia --project=.`) and import `nonKam`.
