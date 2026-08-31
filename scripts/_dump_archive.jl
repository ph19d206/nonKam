#!/usr/bin/env julia
using JLD2

function dump(path)
    jldopen(path, "r") do f
        v = f["datasets"]
        println("datasets type: ", typeof(v), " eltype: ", eltype(v))
        println("num entries: ", length(v))
        for (i, e) in enumerate(v)
            println("\n=== entry $i : ", typeof(e), " ===")
            if e isa NamedTuple || e isa AbstractDict
                for (kk, vv) in pairs(e)
                    if vv isa AbstractVector
                        println("    $kk :: Vector ", length(vv),
                            " head ", vv[1:min(4, end)])
                    elseif vv isa NamedTuple || vv isa AbstractDict
                        println("    $kk = ", vv)
                    else
                        println("    $kk = ", vv)
                    end
                end
            else
                println("    (scalar/other) ", e)
            end
        end
    end
end

dump(ARGS[1])