# src/number_op.jl
"""
    number_op(N::Int; T::Type=Float64)

Constructs the number operator N̂ = diag(0, 1, ..., N-1) in an N-dimensional Fock basis.
Returns a `Diagonal` matrix of size (N, N).
"""
function number_op(N::Int; T::Type=Float64)
    return Diagonal(T.(0:N-1))
end