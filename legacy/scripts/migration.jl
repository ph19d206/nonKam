# migration.jl
using JLD2

# Path to your archive file
archive_file = joinpath(pwd(), "data", "qfi_simulations.jld2")

if isfile(archive_file)
    println("🔄 Opening archive for key migration...")
    
    # Open in read-write mode ("r+")
    jldopen(archive_file, "r+") do file
        # Get a list of all existing keys
        all_keys = keys(file)
        
        for old_key in all_keys
            # Check if the key matches your old pattern (starts with K instead of N)
            if startswith(old_key, "K")
                # Define the new key with N20000 prepended
                new_key = "N20000_" * old_key
                
                println("Renaming: $old_key ➡️  $new_key")
                
                # 1. Copy the data to the new key
                file[new_key] = file[old_key]
                
                # 2. Delete the old key safely
                delete!(file, old_key)
            end
        end
    end
    println("✅ Migration complete! All old keys updated.")
else
    println("❌ Archive file not found at: $archive_file")
end