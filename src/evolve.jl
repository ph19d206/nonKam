# src/evolve.jl

"""
    evolve!(Ufree, D_kick, Vkick, ψ, [buf1, buf2])

Update state vector `ψ` in-place from t to t+1.
`Ufree` and `D_kick` MUST be pre-constructed `Diagonal` operators.
"""
function evolve!(
    Ufree::Diagonal, 
    D_kick::Diagonal, 
    Vkick::Matrix, 
    ψ::Vector{ComplexF64}, 
    buf1::Vector{ComplexF64} = similar(ψ), 
    buf2::Vector{ComplexF64} = similar(ψ)
)
    # 1. Unperturbed evolution: O(N)
    mul!(buf1, Ufree, ψ)              

    # 2. Transform to kick eigenbasis: O(N²)
    mul!(buf2, Vkick', buf1)               

    # 3. Apply diagonal kick: O(N)
    mul!(buf1, D_kick, buf2)           

    # 4. Transform back: O(N²)
    mul!(ψ, Vkick, buf1)
    
    return ψ
end