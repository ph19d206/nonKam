# Legacy research material

This directory preserves superseded exploratory Julia files. They are not part
of the supported package or automated workflow.

## Contents

- `scripts/`: early one-off QFI and mean-energy plotting/running scripts. Use
  the maintained scripts in `../../scripts/` instead.
- `notebooks/`: exported Pluto notebooks. `nonKamParm_nb.jl` also contains a
  machine-specific include path and is retained only for reference.
- `plot-scripts/`: parameter-specific plot scripts tied to older JLD2 archive
  layouts.

`migration.jl` was a one-time in-place data migration for an old
`qfi_simulations.jld2` key format. Do not run it on current data.

The former `src/qfi_slope_r_fidelity.jl` was an empty, unexported stub and was
removed rather than archived.
