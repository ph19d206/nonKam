# Quick smoke test for scripts/plot_qfi_vs_time_at_R.jl (do not commit).
# Runs the script body inside a function so repeated includes are safe.
Base.include(Main, "scripts/plot_qfi_vs_time_at_R.jl")
println("qfi_vs_time plot test: OK")