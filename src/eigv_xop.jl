# eigv_xop

"""
    eigv_xop(N)

Compute the X operator interms of diagonalization. Returns the eigenvalues and eigenvectors of the X operator in the Fock basis. Also "ω" is not there. So here X = (a + a^†)/sqrt(2). 
Also, save the matrices, next time, if same N appear, it will load the matrices from the file instead of computing them again. This is useful for large N, where computing the eigenvalues and eigenvectors can be time-consuming. Caches results inside a local `.cache/` directory to prevent workspace clutter.
"""
function eigv_xop(N::Int)
    # 1. Define a normal, visible cache directory
    cache_dir = joinpath(pwd(), "cache")
    
    # 2. Create the 'cache' folder if it doesn't exist yet
    if !ispath(cache_dir)
        mkdir(cache_dir)
    end
    
    filename = joinpath(cache_dir, "eigv_xop_N$(N).jld2")
    
    if N >= 5000
        # 3. Check cache and load
        if isfile(filename)
            println("Loading matrix from cache folder: cache/eigv_xop_N$(N).jld2")
            data = jldopen(filename, "r") do file
                return file["λx"]::Vector{Float64}, file["V"]::Matrix{Float64}
            end
            return data
        else
            println("Cache miss. Computing eigenvalues and eigenvectors for N=$N...")
            
            # X = (a + a^†) / sqrt(2)
            off_diagonal = sqrt.(1:N-1) ./ sqrt(2)
            zero_diagonal = zeros(N)
            Xop = SymTridiagonal(zero_diagonal, off_diagonal)
            
            λx, V = eigen(Xop)
            
            # 4. Save to the cache directory cleanly
            jldopen(filename, "w") do file
                file["λx"] = λx
                file["V"] = V
            end
            
            return λx, V
        end
    else
        println("Cache miss. Computing eigenvalues and eigenvectors for N=$N...")
        # For smaller N, compute directly without caching
        off_diagonal = sqrt.(1:N-1) ./ sqrt(2)
        zero_diagonal = zeros(N)
        Xop = SymTridiagonal(zero_diagonal, off_diagonal)

        λx, V = eigen(Xop)
        return λx, V
    end
end