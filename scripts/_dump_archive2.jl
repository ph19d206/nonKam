#!/usr/bin/env julia
using JLD2

function dump(path, wantkey=nothing)
    jldopen(path, "r") do f
        ks = collect(keys(f))
        for k in ks
            if wantkey !== nothing && !Base.occursin(wantkey, k)
                continue
            end
            v = f[k]
            println("=== key '$k' : ", typeof(v), " ===")
            if v isa NamedTuple || v isa AbstractDict
                for (kk, vv) in pairs(v)
                    if vv isa AbstractVector
                        println("    $kk :: Vector ", length(vv),
                            " head ", vv[1:min(4, end)])
                    elseif vv isa NamedTuple || vv isa AbstractDict
                        println("    $kk = ", collect(keys(vv)))
                    elseif vv isa Tuple
                        println("    $kk = tuple ", vv)
                    else
                        println("    $kk = ", vv)
                    end
                end
            else
                println("  (other) ", typeof(v))
            end
        end
    end
end

wantkey = length(ARGS) > 1 ? ARGS[2] : nothing
dump(ARGS[1], wantkey)