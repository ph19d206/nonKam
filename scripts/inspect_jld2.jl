#!/usr/bin/env julia

"""
Inspect the structure of a JLD2 data file stored in the project's `data/`
folder: its top-level keys, and (recursively) the type and shape of every
element, so you can see at a glance what a saved file contains.

Usage (from the repository root):

    julia --project=. scripts/inspect_jld2.jl
    julia --project=. scripts/inspect_jld2.jl mean_energy_R_sweep.jld2
    julia --project=. scripts/inspect_jld2.jl path/to/any/file.jld2
    julia --project=. scripts/inspect_jld2.jl mean_energy_R_sweep.jld2 --summary

Environment variables:
    DATA_DIR     directory to look in for bare filenames
                 default: <repo>/data

Within the tree it prints indented lines like:

    datasets :: Vector (1 entry)  of NamedTuple
        entry[1] of NamedTuple{
          signature  :: Tuple (5 elements)
          N          :: Int64 = 1000
          K          :: Float64 = 0.5
          omega      :: Float64 = 0.6366197723675814
          runs       :: Vector (15 entries) of NamedTuple
              entry[1] of NamedTuple{
                R          :: Float64 = 2.0
                ...
          parameters :: NamedTuple{
            ...
"""
function env_s(name, default)
    get(ENV, name, default)
end

using JLD2

"""
Describe a scalar value compactly.
"""
function scalar_desc(x)
    if x isa AbstractString
        startswith(x, "\n") ? repr(x) : repr(x)
    else
        repr(x)
    end
end

"""
Recursively describe `x` at the given indentation, printing each level.

`_maxdepth` guards against pathological recursion (unbounded nested types).
"""
function describe(x, indent::AbstractString=""; _maxdepth::Int=30)
    pad(x) = indent * x
    if _maxdepth <= 0
        println(pad("... (depth limit reached)"))
        return
    end
    if x isa AbstractArray
        if ndims(x) == 0
            println(pad("$(typeof(x)) = $(scalar_desc(x[]))"))
        else
            el = eltype(x)
            println(pad("$(typeof(x)) :: size $(size(x)), eltype $el"))
            # For numeric / string arrays we don't dump every element; for small
            # arrays of scalars show them.
            if length(x) <= 8 && el <: Union{Number, AbstractString}
                println(pad("    values: $(join(map(y -> string(y), vec(x)), ", "))"))
            end
        end
    elseif x isa NamedTuple
        println(pad("$(typeof(x))"))
        for k in keys(x)
            v = getproperty(x, k)
            println(pad("    $k :: $(summarize_type(v))"))
            if !(v isa AbstractArray{<:Number}) && !(v isa AbstractString)
                describe(v, indent * "        "; _maxdepth=_maxdepth - 1)
            end
        end
    elseif x isa Tuple
        println(pad("$(typeof(x)) :: $(length(x)) element(s)"))
        for (i, v) in enumerate(x)
            println(pad("    [$i] :: $(summarize_type(v))"))
            if !(v isa AbstractArray{<:Number}) && !(v isa AbstractString)
                describe(v, indent * "        "; _maxdepth=_maxdepth - 1)
            end
        end
    elseif x isa Dict
        println(pad("$(typeof(x)) :: $(length(x)) key(s)"))
        for (k, v) in x
            println(pad("    $k :: $(summarize_type(v))"))
            if !(v isa AbstractArray{<:Number}) && !(v isa AbstractString)
                describe(v, indent * "        "; _maxdepth=_maxdepth - 1)
            end
        end
    elseif x isa AbstractVector
        println(pad("$(typeof(x)) :: $(length(x)) element(s), eltype $(eltype(x))"))
    else
        println(pad("$(typeof(x)) = $(scalar_desc(x))"))
    end
end

"One-line type summary for use as a field annotation."
function summarize_type(v)
    if v isa AbstractArray
        return "$(typeof(v)) size $(size(v))"
    elseif v isa NamedTuple
        return "NamedTuple {$(join(keys(v), ", "))}"
    elseif v isa Tuple
        return "Tuple :: $(length(v)) elements"
    elseif v isa Dict
        return "Dict :: $(length(v)) keys"
    return string(typeof(v))
    end
end

# --------------------------------------------------------------------------
# Choose the file to inspect.
# --------------------------------------------------------------------------
data_dir_default = joinpath(@__DIR__, "..", "data")
data_dir = get(ENV, "DATA_DIR", data_dir_default)

if length(ARGS) >= 1 && !startswith(ARGS[1], "-")
    fname = ARGS[1]
else
    files = sort(readdir(data_dir; join=true))
    isempty(files) && error("no .jld2 files found in $data_dir")
    if length(files) == 1
        fname = files[1]
    else
        println("No file given. Available JLD2 files in $data_dir:")
        for (i, f) in enumerate(files)
            println("  [$i] $(basename(f))")
        end
        print("Enter a number or filename [1]: ")
        ans = strip(readline())
        if isempty(ans)
            ans = "1"
        end
        if all(isdigit, ans)
            fname = files[parse(Int, ans)]
        else
            fname = joinpath(data_dir, ans)
        end
    end
end

# Bare filenames (and relative paths) are resolved against the data folder.
if !isfile(fname) && !startswith(fname, "/")
    candidate = joinpath(data_dir, fname)
    isfile(candidate) && (fname = candidate)
end

isfile(fname) || error("file not found: $fname")
summary_only = any(a -> a in ("--summary", "-s"), ARGS)

println("Inspecting: $(normpath(fname))")
if summary_only
    jldopen(fname, "r") do f
        println("Top-level groups/keys:")
        for k in keys(f)
            v = f[k]
            println("  $k :: $(summarize_type(v))")
        end
    end
else
    jldopen(fname, "r") do f
        println("Top-level groups/keys:")
        for k in keys(f)
            println("  $k :: $(summarize_type(f[k]))")
        end
        println("\nDetailed structure:")
        for k in keys(f)
            println("$k :: $(summarize_type(f[k]))")
            describe(f[k], "    ")
        end
    end
end
