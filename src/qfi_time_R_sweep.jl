"""
    save_qfi_R_sweep(filepath, sweep, N, ω, K, ϵ, t_max, state)

Write a `floquet_qfi_R_sweep` result to a JLD2 file.  The file has the
following top-level datasets:

- `qfi`: rows are Floquet time steps and columns follow `R`.
- `R`, `tau`, `time_steps`, and `times`: sweep coordinates.
- `parameters`: all fixed simulation parameters and the initial state.

Existing files at `filepath` are replaced deliberately, so one output file is
an exact record of one sweep.
"""
function save_qfi_R_sweep(
    filepath::AbstractString, sweep::NamedTuple, N::Int, ω::Real, K::Real,
    ϵ::Real, t_max::Int, state,
)
    required_fields = (:R, :tau, :time_steps, :times, :qfi)
    all(field -> hasproperty(sweep, field), required_fields) ||
        throw(ArgumentError("sweep must be the result of floquet_qfi_R_sweep."))

    mkpath(dirname(filepath))
    parameters = (
        N=N,
        omega=Float64(ω),
        K=Float64(K),
        epsilon=Float64(ϵ),
        t_max=t_max,
        state=state,
        resonance_relation="omega * tau = 2pi / R",
        qfi_layout="qfi[time_step, R_index]",
    )

    jldopen(filepath, "w") do file
        file["qfi"] = sweep.qfi
        file["R"] = sweep.R
        file["tau"] = sweep.tau
        file["time_steps"] = sweep.time_steps
        file["times"] = sweep.times
        file["parameters"] = parameters
    end
    return filepath
end
