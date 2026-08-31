module nonKam

# import packages
using LinearAlgebra
using JLD2
using Statistics
BLAS.set_num_threads(Threads.nthreads())

# greet() = print("Hello World!")

# include files
include("ClassicalMap.jl")
include("KickedHO_floquet.jl")
include("qfi_with_time_provided_U_a.jl")
include("coherent_state.jl")
include("mean_energy_with_time_for_U_a.jl")
include("calculate_qfi_fidility.jl")
include("save_qfi_fidelity.jl")
include("eigv_xop.jl")
include("eigv_kikU.jl")
include("free_U.jl")
include("floquet_qfi.jl")
include("qfi_time_R_sweep.jl")
include("AsinAop.jl")
include("qfi_generator_evolution.jl")
include("find_converged_m.jl")
include("qfi_scaling_exponent.jl")
include("qfi_scaling_vs_R.jl")
include("grid_around_integer.jl")
include("grid_with_integers.jl")
include("number_op.jl")

# --- added with more care. ---
include("evolve.jl")
include("mean_n.jl")
include("var_n.jl")

# measurement subfolder
include("measurement/quadrature_distribution.jl")
include("measurement/quadrature_derivative.jl")
include("measurement/quadrature_cfi.jl")

# plotting helper
include("plot_helper.jl")

# export functions
export ClassicalMap, KickedHO_floquet, qfi_with_time_provided_U_a
export coherent_state, mean_energy_with_time_for_U_a
export calculate_qfi_fidility, save_qfi_fidelity!
export floquet_qfi, floquet_qfi_R_sweep, save_qfi_R_sweep
export AsinAop, qfi_generator_evolution
export find_converged_m, find_power_law_regimes
export qfi_scaling_exponent, qfi_scaling_vs_R
export grid_around_integer, grid_with_integers
export number_op

# --- added with care ---
export eigv_kikU
export free_U
export evolve!
export mean_n
export var_n
export quadrature_distribution
export quadrature_derivative
export quadrature_cfi
export plot_curves

end # module nonKam
